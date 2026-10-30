import glob
import os
import re
import shutil
import sys

REL = "mlx_lm/server.py"
VERSION_REL = "mlx_lm/_version.py"
EXPECTED_VERSION = "0.31.3"

TAG = "# PATCH (crucible 2026-10-29, envs/llm/patches/patch_mlx_lm_tool_call_ends_reasoning.py)"

# Stock mlx-lm 0.31.3 leaves its "reasoning" state only on </think>. A Qwen that writes its
# tool call before closing its thinking (qwen3.5-9b, round two of a tool loop with thinking
# on: bookforge pc 1, 2026-10-29, 2 of 2 runs) had the whole <tool_call> block read as
# thinking: finish "stop", no tool_calls, empty content, the call as text at the end of
# `reasoning`. vLLM's qwen3 reasoning parser ends the reasoning at <tool_call>; patched,
# mlx-lm does too, so the call is parsed as a call. Only when the request sent tools (the
# tool state exists only then, mlx-lm-tools-only-when-sent).
EDITS = (
    (
        "            sequences[ts] = tokenizer.tool_call_start\n"
        "            if te:\n"
        "                sequences[te] = tokenizer.tool_call_end\n",
        "            sequences[ts] = tokenizer.tool_call_start\n"
        "            if te:\n"
        "                sequences[te] = tokenizer.tool_call_end\n"
        "            " + TAG + ": a tool call written while thinking ends the thinking.\n"
        "            if tokenizer.has_thinking:\n"
        "                transitions[\"reasoning\"].append((ts, \"tool\"))\n",
    ),
)

MARKER = 'transitions["reasoning"].append((ts, "tool"))'


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
            "usage: patch_mlx_lm_tool_call_ends_reasoning.py <env-prefix>   "
            "(or set CRUCIBLE_LLM_ENV)"
        )
    path = site_packages_file(prefix, REL)
    with open(path, encoding="utf-8", newline="") as handle:
        live = handle.read()

    if MARKER in live:
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
    if MARKER not in patched:
        print(f"INCOMPLETE: a tool call still cannot end the reasoning in {path}", file=sys.stderr)
        sys.exit(2)

    shutil.copy2(path, path + ".orig-tool-in-reasoning")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(patched)
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
