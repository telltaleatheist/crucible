import glob
import os
import re
import shutil
import sys

REL = "mlx_lm/server.py"
VERSION_REL = "mlx_lm/_version.py"
EXPECTED_VERSION = "0.31.3"

TAG = "# PATCH (crucible 2026-10-25, envs/llm/patches/patch_mlx_lm_tools_only_when_sent.py)"

# Stock mlx-lm 0.31.3 enters its "tool" state on the model's tool-call token whatever the
# request sent. With no `tools`, its parser is given none, fails, and the formatter drops
# the call: the answer came back `finish_reason: "tool_calls"` with neither content nor
# tool_calls (Briefcase on the Mac, qwen3.5-9b, 2026-10-25), so the model's output was
# lost. A request that sent no tools gets the token as text, as vLLM and llama-server do.
EDITS = (
    (
        '    def _make_state_machine(\n'
        '        self, model_key, tokenizer, stop_words, initial_state="normal"\n'
        '    ):\n',
        '    def _make_state_machine(\n'
        '        self, model_key, tokenizer, stop_words, initial_state="normal", tools=False\n'
        '    ):\n',
    ),
    (
        "        cache_key = (model_key, tuple(stop_words), initial_state)\n",
        "        " + TAG + ": a request with no tools reads no tool calls.\n"
        "        cache_key = (model_key, tuple(stop_words), initial_state, bool(tools))\n",
    ),
    (
        "        if tokenizer.has_tool_calling:\n"
        "            ts = tokenizer.tool_call_start_tokens\n",
        "        if tokenizer.has_tool_calling and tools:\n"
        "            ts = tokenizer.tool_call_start_tokens\n",
    ),
    (
        "                        args.stop_words,\n"
        "                        initial_state,\n"
        "                    )\n",
        "                        args.stop_words,\n"
        "                        initial_state,\n"
        "                        tools=bool(request.tools),\n"
        "                    )\n",
    ),
    (
        "                args.stop_words,\n"
        "                initial_state=initial_state,\n"
        "            )\n",
        "                args.stop_words,\n"
        "                initial_state=initial_state,\n"
        "                tools=bool(request.tools),\n"
        "            )\n",
    ),
)

MARKER = "cache_key = (model_key, tuple(stop_words), initial_state, bool(tools))"

ABSENT_MARKER = "cache_key = (model_key, tuple(stop_words), initial_state)\n"


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
            "usage: patch_mlx_lm_tools_only_when_sent.py <env-prefix>   "
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
    if MARKER not in patched or ABSENT_MARKER in patched:
        print(f"INCOMPLETE: the tool state is still unconditional in {path}", file=sys.stderr)
        sys.exit(2)

    shutil.copy2(path, path + ".orig-tools")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(patched)
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
