"""Reading the GGUFs Ollama already has on this machine.

Owen, 2026-09-16: *"if its possible to use the ollama copies that already exist
on disk then we should do that. i dont want to have 16 copies of giant models
sitting around"*, and *"windows crucible should function if there's no wsl. It
won't have batching but it should work. It should be able to use the ollama
models or api keys via Bookforge/foundry if the user wants to do that instead"*.

WHICH BACKEND THIS IS FOR, AND WHY ONLY ONE. Measured on Owen's PC 2026-09-16:
`~/.ollama/models` is 64 GB and its largest blob's first four bytes are
`47 47 55 46` — `GGUF`. Ollama stores GGUF and nothing else. llama.cpp reads GGUF
and takes any path for `-m`, so `llama-windows` can serve straight out of that
store. vLLM and mlx-lm want safetensors and cannot, at any price short of
re-quantizing. So this module is `llama-windows`'s alone, and a `cuda-linux` host
saves nothing by having Ollama installed.

THE BYTES ARE NOT THE MANIFEST'S BYTES, AND THAT IS THE WHOLE DESIGN PROBLEM.
A `[backends.llama-windows]` block pins a HuggingFace repo, a revision and a
file — `unsloth/Qwen3.5-9B-GGUF` @ 3885219b, `Qwen3.5-9B-Q8_0.gguf`. The
`[local]` table names a different thing: `qwen3.5:9b-bf16`, published by Ollama,
quantized by somebody else at a different precision. They are two forms of one
model and they are NOT the same file.

So serving Ollama's blob under the manifest's `fingerprint` would be exactly the
silent substitution `weights.installed()` refuses when a stamp names a different
revision — and worse, because a fingerprint is a RECORD: Foundry hashes it into
its cleanup cache key and BookForge stamps it into a book's OPF
(CLIENT-SURFACES.md 6.5). Two different quantizations filed under one name means
a cache that answers for weights that never ran.

The answer is not to refuse the reuse, it is to SAY WHICH ONE RAN. Weights found
here are a source of their own, with their own provenance
(`ollama:<tag>@sha256:<digest>`), and nothing pretends they came from the hub.
That is the same discipline `[local] needs_basis` already keeps: a number that
was declared rather than measured travels as `declared` all the way to the
screen.

WHAT IS CHECKED BEFORE A BLOB IS OFFERED. Three things, and each one has a
failure it is there for:

1. The manifest for the tag exists and names a layer of mediaType
   `application/vnd.ollama.image.model`. A tag whose model layer is absent is a
   partial pull.
2. The blob file exists at `blobs/sha256-<digest>`. Ollama writes the manifest
   last, but a hand-deleted blob leaves a manifest pointing at nothing.
3. The blob's size on disk equals the size the manifest's layer declares. A
   truncated download is the one failure that otherwise reaches llama.cpp as a
   parse error three layers away from the file that was wrong.

The digest is NOT verified by hashing: it is up to 19 GB and this runs on the
path to a model load. The size check is what is affordable; a caller that wants
certainty has the digest in hand and can spend the minutes itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import CrucibleError

#: Where Ollama keeps its store, and the environment variable that moves it.
#: `OLLAMA_MODELS` is Ollama's own name for this and is honoured for the reason
#: every other path in this repo is read rather than assumed: Owen's C: drive
#: has filled twice, and a store moved to another drive is exactly what somebody
#: does about that.
STORE_ENV = "OLLAMA_MODELS"

#: The registry a bare tag belongs to. Ollama writes
#: `manifests/<registry>/<namespace>/<name>/<tag>`, and a tag written the short
#: way — `qwen3.5:9b-bf16`, which is what the `[local]` tables carry — means
#: this registry and this namespace.
DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"

#: The layer that is the weights. Ollama's other layers are the licence, the
#: parameters and the template, and a model layer is the only one that is a
#: GGUF.
MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"

#: The projector, for a model that reads images. Same store, separate layer —
#: `dots-ocr`'s llama-windows block needs both, and half a vision model is a
#: model that loads and then cannot see (PHASE15-HOST.md 3.10, fact 2).
PROJECTOR_MEDIA_TYPE = "application/vnd.ollama.image.projector"


class OllamaStoreError(CrucibleError):
    """A tag was asked for and the store could not honestly answer."""


@dataclass(frozen=True)
class OllamaBlob:
    """One file in Ollama's store, and where it came from."""

    path: Path
    digest: str
    bytes: int
    media_type: str


@dataclass(frozen=True)
class OllamaWeights:
    """A tag's weights as this machine holds them.

    `provenance` is what a job's sidecar records instead of the hub pin, so a
    run served from here can never be mistaken afterwards for a run served from
    the file the manifest pins.
    """

    tag: str
    model: OllamaBlob
    projector: OllamaBlob | None
    root: Path

    @property
    def provenance(self) -> str:
        return f"ollama:{self.tag}@{self.model.digest}"

    @property
    def bytes(self) -> int:
        total = self.model.bytes
        if self.projector is not None:
            total += self.projector.bytes
        return total


def store_root(
    environ: Mapping[str, str] | None = None, home: Path | None = None
) -> Path:
    """Where this machine's Ollama store is, whether or not it exists.

    Returns a path rather than None for a missing store, because "there is no
    Ollama here" and "Ollama is here and does not have that tag" are different
    answers and the caller needs to tell them apart — `present()` is the first
    question and `resident()` is the second.
    """
    env = os.environ if environ is None else environ
    override = env.get(STORE_ENV)
    if override:
        return Path(override)
    return (Path.home() if home is None else home) / ".ollama" / "models"


def present(root: Path) -> bool:
    """Is there an Ollama store here at all."""
    return (root / "manifests").is_dir() and (root / "blobs").is_dir()


def split_tag(tag: str) -> tuple[str, str, str, str]:
    """`qwen3.5:9b-bf16` -> (registry, namespace, name, tag).

    Accepts the long forms Ollama also writes — `library/qwen3.5:9b-bf16` and
    `registry.ollama.ai/library/qwen3.5:9b-bf16` — because a `[local]` table is
    allowed to be explicit and a reader should not have to know that the short
    form is the only one that works.
    """
    if not tag or tag.count(":") != 1:
        raise OllamaStoreError(
            f"{tag!r} is not an Ollama tag; a tag is <name>:<label>, "
            "optionally with a registry and namespace in front of it"
        )
    path, _, label = tag.partition(":")
    parts = [part for part in path.split("/") if part]
    if len(parts) == 1:
        return DEFAULT_REGISTRY, DEFAULT_NAMESPACE, parts[0], label
    if len(parts) == 2:
        return DEFAULT_REGISTRY, parts[0], parts[1], label
    if len(parts) == 3:
        return parts[0], parts[1], parts[2], label
    raise OllamaStoreError(
        f"{tag!r} has {len(parts)} path segments; an Ollama tag has at most "
        "three (registry/namespace/name)"
    )


def manifest_path(root: Path, tag: str) -> Path:
    registry, namespace, name, label = split_tag(tag)
    return root / "manifests" / registry / namespace / name / label


def blob_path(root: Path, digest: str) -> Path:
    """`sha256:abc…` -> `<root>/blobs/sha256-abc…`, which is Ollama's spelling."""
    return root / "blobs" / digest.replace(":", "-", 1)


def _layer(
    root: Path, tag: str, layers: list[dict], media_type: str
) -> OllamaBlob | None:
    for layer in layers:
        if layer.get("mediaType") != media_type:
            continue
        digest = layer.get("digest")
        declared = layer.get("size")
        if not isinstance(digest, str) or not isinstance(declared, int):
            raise OllamaStoreError(
                f"{tag}: the {media_type} layer states digest {digest!r} and "
                f"size {declared!r}; both are required to find and check a blob"
            )
        path = blob_path(root, digest)
        if not path.is_file():
            raise OllamaStoreError(
                f"{tag}: the manifest names {digest} and {path} is not there. "
                "The store has a manifest for a blob it does not hold — pull the "
                "tag again with `ollama pull`, or let Crucible fetch its own copy"
            )
        found = path.stat().st_size
        if found != declared:
            raise OllamaStoreError(
                f"{tag}: {path.name} is {found} bytes and the manifest says "
                f"{declared}. A truncated blob reaches llama.cpp as a parse "
                "error several layers from the file that is wrong, so it is "
                "refused here instead"
            )
        return OllamaBlob(
            path=path, digest=digest, bytes=declared, media_type=media_type
        )
    return None


def resident(root: Path, tag: str, *, wants_projector: bool = False) -> OllamaWeights:
    """This tag's weights in this store, or a refusal that names what is wrong.

    Raises rather than returning None, because every way of not finding a tag is
    a different thing for a caller to do about it — no store, no manifest for
    this tag, a manifest with no model layer, a blob that is absent or the wrong
    size — and a None would collapse all five into "pull it again".
    """
    if not present(root):
        raise OllamaStoreError(
            f"there is no Ollama store at {root}; set ${STORE_ENV} if it lives "
            "somewhere else"
        )
    path = manifest_path(root, tag)
    if not path.is_file():
        raise OllamaStoreError(
            f"{tag} is not in this Ollama store ({path} is not there). "
            f"`ollama pull {tag}` would put it there"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OllamaStoreError(f"{tag}: {path} is not readable JSON: {exc}") from None
    layers = document.get("layers")
    if not isinstance(layers, list):
        raise OllamaStoreError(
            f"{tag}: {path} has no `layers` array; this is not an Ollama manifest"
        )
    model = _layer(root, tag, layers, MODEL_MEDIA_TYPE)
    if model is None:
        kinds = sorted({str(layer.get("mediaType")) for layer in layers})
        raise OllamaStoreError(
            f"{tag}: the manifest has no {MODEL_MEDIA_TYPE} layer, only "
            f"{kinds}. A tag with no model layer is a partial pull"
        )
    projector = _layer(root, tag, layers, PROJECTOR_MEDIA_TYPE)
    if wants_projector and projector is None:
        raise OllamaStoreError(
            f"{tag}: this model reads images and the tag has no "
            f"{PROJECTOR_MEDIA_TYPE} layer. Half a vision model is a model that "
            "loads and then cannot see (PHASE15-HOST.md 3.10, fact 2)"
        )
    return OllamaWeights(tag=tag, model=model, projector=projector, root=root)


def held(root: Path) -> tuple[str, ...]:
    """Every tag this store holds, as `<name>:<label>`, sorted.

    For a settings page that offers what is already on disk before it offers a
    download. Only the default registry and namespace are spelled short; anything
    else is spelled in full, so a tag from this list can be handed straight back
    to `resident()`.
    """
    manifests = root / "manifests"
    if not manifests.is_dir():
        return ()
    found: list[str] = []
    for label in manifests.rglob("*"):
        if not label.is_file():
            continue
        parts = label.relative_to(manifests).parts
        if len(parts) != 4:
            continue
        registry, namespace, name, tag = parts
        if registry == DEFAULT_REGISTRY and namespace == DEFAULT_NAMESPACE:
            found.append(f"{name}:{tag}")
        else:
            found.append(f"{registry}/{namespace}/{name}:{tag}")
    return tuple(sorted(found))
