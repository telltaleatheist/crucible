import glob
import os
import re
import shutil
import sys

REL = "mlx_lm/generate.py"
VERSION_REL = "mlx_lm/_version.py"
EXPECTED_VERSION = "0.31.3"

TAG = "# PATCH (crucible 2026-09-26, envs/llm/patches/patch_mlx_lm_cache_counters.py)"

HELPER_ANCHOR = "\n\nclass GenerationBatch:\n"
HELPER = (
    "\n\n" + TAG + ":\n"
    "# the caches' bookkeeping arrays, so the decode step can evaluate them. Stock\n"
    "# 0.31.3 updates them lazily and never forces them, and a hybrid model's\n"
    "# graph then grows until Metal's buffer count limit (499000) kills the thread.\n"
    "_CRUCIBLE_CACHE_COUNTERS = (\"left_padding\", \"lengths\", \"offset\", \"_idx\")\n"
    "\n"
    "\n"
    "def _crucible_cache_counters(caches, out=None):\n"
    "    out = [] if out is None else out\n"
    "    for c in caches or ():\n"
    "        if c is None:\n"
    "            continue\n"
    "        inner = getattr(c, \"caches\", None)\n"
    "        if isinstance(inner, (list, tuple)):\n"
    "            _crucible_cache_counters(inner, out)\n"
    "        for name in _CRUCIBLE_CACHE_COUNTERS:\n"
    "            value = getattr(c, name, None)\n"
    "            if isinstance(value, mx.array):\n"
    "                out.append(value)\n"
    "    return out\n"
    "\n\nclass GenerationBatch:\n"
)

EDITS = (
    (
        "        mx.async_eval(self._next_tokens, self._next_logprobs, token_context)\n",
        "        " + TAG + "\n"
        "        mx.async_eval(\n"
        "            self._next_tokens,\n"
        "            self._next_logprobs,\n"
        "            token_context,\n"
        "            _crucible_cache_counters(self.prompt_cache),\n"
        "        )\n",
    ),
)

MARKER = "_crucible_cache_counters(self.prompt_cache),"

ABSENT_MARKER = "mx.async_eval(self._next_tokens, self._next_logprobs, token_context)"


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


def main() -> None:
    prefix = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CRUCIBLE_LLM_ENV", "")
    ).rstrip("/")
    if not prefix:
        raise SystemExit(
            "usage: patch_mlx_lm_cache_counters.py <env-prefix>   (or set CRUCIBLE_LLM_ENV)"
        )
    path = site_packages_file(prefix, REL)
    with open(path, encoding="utf-8", newline="") as handle:
        live = handle.read()

    if MARKER in live and ABSENT_MARKER not in live:
        print("ALREADY_PATCHED " + path)
        return

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

    anchors = [(HELPER_ANCHOR, HELPER)] + list(EDITS)
    for old, _ in anchors:
        if live.count(old) != 1:
            print(
                f"ANCHOR_NOT_FOUND: expected exactly one {old.strip()!r} in {path}, "
                f"found {live.count(old)}",
                file=sys.stderr,
            )
            sys.exit(2)

    patched = live
    for old, new in anchors:
        patched = patched.replace(old, new)
    if MARKER not in patched or ABSENT_MARKER in patched:
        print(f"INCOMPLETE: the stock async_eval survived in {path}", file=sys.stderr)
        sys.exit(2)

    shutil.copy2(path, path + ".orig")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(patched)
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
