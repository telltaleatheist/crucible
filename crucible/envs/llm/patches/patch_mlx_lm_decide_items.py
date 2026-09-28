import glob
import os
import re
import shutil
import sys

REL = "mlx_lm/server.py"
VERSION_REL = "mlx_lm/_version.py"
EXPECTED_VERSION = "0.31.3"

TAG = "# PATCH (crucible 2026-09-28, envs/llm/patches/patch_mlx_lm_decide_items.py)"

EDITS = (
    (
        "        request_factories = {\n"
        '            "/v1/completions": self.handle_text_completions,\n',
        "        " + TAG + ":\n"
        "        # the decision door's items form: many questions about one state,\n"
        "        # the state run once; mlx_lm/_crucible_items.py (Crucible's\n"
        "        # engines/items_forward.py, placed by patch_mlx_lm_decide_items_helper.py)\n"
        "        # answers it on the generation thread.\n"
        '        if self.path == "/v1/crucible/items":\n'
        "            from mlx_lm import _crucible_items\n"
        "\n"
        "            _crucible_items.mlx_lm_serve_http(self)\n"
        "            return\n"
        "\n"
        "        request_factories = {\n"
        '            "/v1/completions": self.handle_text_completions,\n',
    ),
    (
        "            # We got a request\n"
        "            if request is not None:\n"
        "                rqueue, request, args = request\n",
        "            # We got a request\n"
        "            if request is not None:\n"
        "                rqueue, request, args = request\n"
        "\n"
        "                " + TAG + ":\n"
        "                # an items pass runs alone: an active batch drains first.\n"
        '                if type(request).__name__ == "MlxLmItemsJob":\n'
        "                    from mlx_lm import _crucible_items\n"
        "\n"
        "                    if batch_generator is not None and len(batch_results) > 0:\n"
        "                        drain_batch = True\n"
        "                        unprocessed_requests.append((rqueue, request, args))\n"
        "                        continue\n"
        "                    _crucible_items.mlx_lm_run_job(self.model_provider, request, rqueue)\n"
        "                    continue\n",
    ),
)

MARKER = "_crucible_items.mlx_lm_run_job(self.model_provider, request, rqueue)"

HTTP_MARKER = "_crucible_items.mlx_lm_serve_http(self)"


def site_packages_file(prefix: str, rel: str) -> str:
    hits = sorted(
        {
            os.path.realpath(p)
            for p in glob.glob(f"{prefix}/lib/python*/site-packages/{rel}")
        }
    )
    if not hits:
        raise SystemExit(f"NOT_FOUND: no {rel} under {prefix}/lib/python*/site-packages")
    if len(hits) > 1:
        raise SystemExit(
            f"AMBIGUOUS: {len(hits)} distinct site-packages trees under {prefix}: {hits}"
        )
    return hits[0]


def check_version(prefix: str) -> None:
    version_path = site_packages_file(prefix, VERSION_REL)
    with open(version_path, encoding="utf-8") as handle:
        found = re.search(r'__version__\s*=\s*"([^"]+)"', handle.read())
    version = found.group(1) if found else None
    if version != EXPECTED_VERSION:
        print(
            f"VERSION_MISMATCH: this patch was derived against mlx-lm "
            f"{EXPECTED_VERSION} and {version_path} says {version!r}; re-derive it",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    prefix = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CRUCIBLE_LLM_ENV", "")
    ).rstrip("/")
    if not prefix:
        raise SystemExit(
            "usage: patch_mlx_lm_decide_items.py <env-prefix>   (or set CRUCIBLE_LLM_ENV)"
        )
    path = site_packages_file(prefix, REL)
    with open(path, encoding="utf-8", newline="") as handle:
        live = handle.read()

    if MARKER in live and HTTP_MARKER in live:
        print("ALREADY_PATCHED " + path)
        return

    check_version(prefix)
    for old, _ in EDITS:
        if live.count(old) != 1:
            print(
                f"ANCHOR_NOT_FOUND: expected exactly one {old.strip()!r} in {path}, "
                f"found {live.count(old)}",
                file=sys.stderr,
            )
            sys.exit(2)

    patched = live
    for old, new in EDITS:
        patched = patched.replace(old, new)

    shutil.copy2(path, path + ".orig")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(patched)
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
