import glob
import os
import shutil
import sys

REL = "mlx_lm/server.py"

OLD = '        self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])'

NEW = (
    "        # PATCH (crucible 2026-09-23, envs/llm/patches/patch_mlx_lm_top_logprobs.py):\n"
    "        # the decide door reads up to 26 options plus a margin of 4; stock\n"
    "        # mlx-lm caps top_logprobs at 11. Nothing downstream assumes 11.\n"
    '        self._validate("top_logprobs", int, min_val=0, max_val=40, whitelist=[-1])'
)

MARKER = 'self._validate("top_logprobs", int, min_val=0, max_val=40, whitelist=[-1])'

ABSENT_MARKER = 'self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])'


def target_path() -> str:
    prefix = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CRUCIBLE_LLM_ENV", "")
    ).rstrip("/")
    if not prefix:
        raise SystemExit(
            "usage: patch_mlx_lm_top_logprobs.py <env-prefix>   (or set CRUCIBLE_LLM_ENV)"
        )
    hits = sorted(
        {
            os.path.realpath(p)
            for p in glob.glob(f"{prefix}/lib/python*/site-packages/{REL}")
        }
    )
    if not hits:
        raise SystemExit(f"NOT_FOUND: no {REL} under {prefix}/lib/python*/site-packages")
    if len(hits) > 1:
        raise SystemExit(
            f"AMBIGUOUS: {len(hits)} distinct site-packages trees under {prefix}: {hits}"
        )
    return hits[0]


def main() -> None:
    path = target_path()
    with open(path, encoding="utf-8", newline="") as handle:
        live = handle.read()

    if MARKER in live and ABSENT_MARKER not in live:
        print("ALREADY_PATCHED " + path)
        return

    if live.count(OLD) != 1:
        print(
            f"ANCHOR_NOT_FOUND: expected exactly one {OLD.strip()!r} in {path}, "
            f"found {live.count(OLD)}",
            file=sys.stderr,
        )
        sys.exit(2)

    shutil.copy2(path, path + ".orig")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(live.replace(OLD, NEW))
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
