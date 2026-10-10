"""Optional packages: models a server holds only when it is installed.

Owen, 2026-10-10: embed and rerank are "an OPTIONAL install package, like the voice/Higgs
package: a machine that doesn't install it (Victoria's 8 GiB laptop) never gets these
models". A package is named in classnames.PACKAGE_NAMES; a model is in it by what its
manifest is (ModelManifest.package: an [embed] or [rerank] table is `retrieval`); a server
has it when its config says `[packages] <name> = true`, which `crucible install <name>`
writes after pulling the models. Where it is not installed, every door that would load or
pick one of its models refuses by name (`package_not_installed`), install-on-submit
included, so nothing pulls them by itself.
"""

from __future__ import annotations

from typing import Any

from .classnames import PACKAGE_NAMES, RETRIEVAL_PACKAGE
from .errors import ApiError

PURPOSE: dict[str, str] = {
    RETRIEVAL_PACKAGE: (
        "the embed and rerank verbs' own models (Qwen3-Embedding-8B and "
        "Qwen3-Reranker-8B, bf16, about 16 GB each)"
    ),
}


def package_not_installed(model: str, package: str, doing: str) -> ApiError:
    """`model` is in an optional package this server has not installed, so it does not
    `doing` ("load it", "embed text with it"): by name, with the command that installs it."""
    return ApiError(
        409,
        "package_not_installed",
        f"{model!r} is in the optional {package} package, which this server has not "
        f"installed, so it does not {doing}. `crucible install {package}` on the server "
        "pulls the package's models and turns it on; a server without it never holds them",
        {"model": model, "package": package},
    )


def models_of(package: str, manifests: dict[str, Any], backend_kind: str) -> list[Any]:
    """The package's models that have a block on this backend, in id order."""
    if package not in PACKAGE_NAMES:
        raise KeyError(f"{package!r} is not a package; this build's are {list(PACKAGE_NAMES)}")
    return [
        manifest
        for _, manifest in sorted(manifests.items())
        if getattr(manifest, "package", None) == package and manifest.supports(backend_kind)
    ]


__all__ = ["PACKAGE_NAMES", "PURPOSE", "models_of", "package_not_installed"]
