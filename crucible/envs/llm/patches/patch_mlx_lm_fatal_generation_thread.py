import glob
import os
import re
import shutil
import sys

REL = "mlx_lm/server.py"
VERSION_REL = "mlx_lm/_version.py"
EXPECTED_VERSION = "0.31.3"

TAG = "# PATCH (crucible 2026-09-26, envs/llm/patches/patch_mlx_lm_fatal_generation_thread.py)"

HELPER_ANCHOR = "\n\nclass ResponseGenerator:\n"
HELPER = (
    "\n\n" + TAG + ":\n"
    "# an exception out of the generation thread exits the PROCESS. Stock, it\n"
    "# ended only the thread, and every accepted request then waited forever on\n"
    "# a queue nothing would fill while the server still answered /v1/models.\n"
    "def _crucible_fatal_thread(target):\n"
    "    def run():\n"
    "        try:\n"
    "            target()\n"
    "        except BaseException:\n"
    "            import os as _os\n"
    "            import sys as _sys\n"
    "            import traceback as _traceback\n"
    "\n"
    "            _traceback.print_exc()\n"
    "            print(\n"
    "                'crucible: the generation thread died; exiting 70 so the '\n"
    "                'engine is seen to be gone',\n"
    "                file=_sys.stderr,\n"
    "            )\n"
    "            _sys.stderr.flush()\n"
    "            _sys.stdout.flush()\n"
    "            _os._exit(70)\n"
    "\n"
    "    return run\n"
    "\n\nclass ResponseGenerator:\n"
)

EDITS = (
    (
        "        self._generation_thread = Thread(target=self._generate)\n",
        "        " + TAG + "\n"
        "        self._generation_thread = Thread(\n"
        "            target=_crucible_fatal_thread(self._generate)\n"
        "        )\n",
    ),
)

MARKER = "target=_crucible_fatal_thread(self._generate)"

ABSENT_MARKER = "Thread(target=self._generate)"


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
            "usage: patch_mlx_lm_fatal_generation_thread.py <env-prefix>   "
            "(or set CRUCIBLE_LLM_ENV)"
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
        print(f"INCOMPLETE: the bare thread survived the edits in {path}", file=sys.stderr)
        sys.exit(2)

    shutil.copy2(path, path + ".orig")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(patched)
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
