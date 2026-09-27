from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import CrucibleError

STORE_ENV = "OLLAMA_MODELS"

DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"

MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"

PROJECTOR_MEDIA_TYPE = "application/vnd.ollama.image.projector"


class OllamaStoreError(CrucibleError):
    ...


@dataclass(frozen=True)
class OllamaBlob:
    path: Path
    digest: str
    bytes: int
    media_type: str


@dataclass(frozen=True)
class OllamaWeights:
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
    env = os.environ if environ is None else environ
    override = env.get(STORE_ENV)
    if override:
        return Path(override)
    return (Path.home() if home is None else home) / ".ollama" / "models"


def present(root: Path) -> bool:
    return (root / "manifests").is_dir() and (root / "blobs").is_dir()


def split_tag(tag: str) -> tuple[str, str, str, str]:
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
            "loads and then cannot see (docs/internals/engines-and-capability.md, \"llama-server\")"
        )
    return OllamaWeights(tag=tag, model=model, projector=projector, root=root)


def held(root: Path) -> tuple[str, ...]:
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
