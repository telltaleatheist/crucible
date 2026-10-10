"""Enforce response_format / structured_outputs in mlx-lm 0.31.3's server.

Stock mlx-lm reads no response_format: a chat that asked for a JSON schema was
answered unconstrained and nobody was told. Patched, the server reads the
constraint on the HTTP thread (mlx_lm/_crucible_grammar.py, Crucible's
engines/structured_mlx.py, placed by patch_mlx_lm_structured_output_helper.py),
refuses by name what it cannot enforce, and gives each constrained sequence an
llguidance logits processor of its own, last in its list.

Two files:

- mlx_lm/server.py: GenerationArguments carries the constraint; do_POST reads it;
  both _make_logits_processors call sites pass the tokenizer and get the grammar
  processor appended; a constrained chat starts in the "normal" state, because the
  grammar binds from the first generated token as it does on vLLM 0.29.0 (no
  reasoning parser), so the answer is content and not reasoning; and a matcher that
  failed during generation answers with the error, not the partial text.
- mlx_lm/generate.py: two stock defects in how a batch carries per-sequence logits
  processors, harmless while every list is empty and fatal or silent once one is
  not. PromptProcessingBatch.extend turns a batch of empty lists into None entries
  (`[None] * len(self.uids)`), so a plain chat and a constrained chat prefilled
  together reach GenerationBatch._step as [None, [grammar]] and `for processor in
  None` kills the generation thread: _step iterates `... or ()`. And
  GenerationBatch.filter filters the list only `if any(...)`, so after a plain
  sequence finishes, a list of empty entries stays longer than uids and the next
  constrained sequence extended in sits past the end: its grammar is never run and
  another row's (empty) list is used in its place, which is an unconstrained
  answer with nothing said. filter now filters any non-empty list.
"""

import glob
import os
import re
import shutil
import sys

REL = "mlx_lm/server.py"
GENERATE_REL = "mlx_lm/generate.py"
VERSION_REL = "mlx_lm/_version.py"
EXPECTED_VERSION = "0.31.3"

TAG = "# PATCH (crucible 2026-10-10, envs/llm/patches/patch_mlx_lm_structured_output.py)"

SERVER_EDITS = (
    (
        "    chat_template_kwargs: Optional[Dict[str, Any]]\n"
        "\n"
        "\n"
        "@dataclass\n"
        "class CompletionRequest:\n",
        "    chat_template_kwargs: Optional[Dict[str, Any]]\n"
        "    " + TAG + ":\n"
        "    # the request's response_format / structured_outputs, compiled to an\n"
        "    # llguidance grammar (mlx_lm/_crucible_grammar.py Constraint), or None.\n"
        "    crucible_constraint: Optional[Any] = None\n"
        "\n"
        "\n"
        "@dataclass\n"
        "class CompletionRequest:\n",
    ),
    (
        "def _make_logits_processors(args):\n"
        "    return make_logits_processors(\n"
        "        args.logits.logit_bias,\n"
        "        args.logits.repetition_penalty,\n"
        "        args.logits.repetition_context_size,\n"
        "        args.logits.presence_penalty,\n"
        "        args.logits.presence_context_size,\n"
        "        args.logits.frequency_penalty,\n"
        "        args.logits.frequency_context_size,\n"
        "    )\n",
        "def _make_logits_processors(args, tokenizer):\n"
        "    processors = make_logits_processors(\n"
        "        args.logits.logit_bias,\n"
        "        args.logits.repetition_penalty,\n"
        "        args.logits.repetition_context_size,\n"
        "        args.logits.presence_penalty,\n"
        "        args.logits.presence_context_size,\n"
        "        args.logits.frequency_penalty,\n"
        "        args.logits.frequency_context_size,\n"
        "    )\n"
        "    " + TAG + ":\n"
        "    # the grammar goes last, so no bias or penalty lifts a token it forbids.\n"
        "    if args.crucible_constraint is not None:\n"
        "        from mlx_lm import _crucible_grammar\n"
        "\n"
        "        processors = [\n"
        "            *processors,\n"
        "            _crucible_grammar.GrammarProcessor(args.crucible_constraint, tokenizer),\n"
        "        ]\n"
        "    return processors\n",
    ),
    (
        "            if think_start > think_end:\n"
        '                initial_state = "reasoning"\n'
        "\n",
        "            if think_start > think_end:\n"
        '                initial_state = "reasoning"\n'
        "\n"
        "        " + TAG + ":\n"
        "        # a constrained answer is bound from its first token, as vLLM binds it\n"
        "        # with no reasoning parser, so what it writes is the answer.\n"
        "        if args.crucible_constraint is not None:\n"
        '            initial_state = "normal"\n'
        "\n",
    ),
    (
        "                        logits_processors=[_make_logits_processors(args)],\n",
        "                        logits_processors=[_make_logits_processors(args, tokenizer)],\n",
    ),
    (
        "            logits_processors = _make_logits_processors(args)\n",
        "            logits_processors = _make_logits_processors(args, tokenizer)\n",
    ),
    (
        '        self.chat_template_kwargs = self.body.get("chat_template_kwargs")\n'
        "        self.validate_model_parameters()\n",
        '        self.chat_template_kwargs = self.body.get("chat_template_kwargs")\n'
        "        self.validate_model_parameters()\n"
        "\n"
        "        " + TAG + ":\n"
        "        # a constraint is enforced with llguidance or refused by name, never\n"
        "        # dropped.\n"
        "        from mlx_lm import _crucible_grammar\n"
        "\n"
        "        try:\n"
        "            self.crucible_constraint = _crucible_grammar.constraint_of_body(self.body)\n"
        "        except _crucible_grammar.GrammarRefusal as refusal:\n"
        "            _crucible_grammar.refuse(self, refusal)\n"
        "            return\n",
    ),
    (
        "            chat_template_kwargs=self.chat_template_kwargs,\n"
        "        )\n",
        "            chat_template_kwargs=self.chat_template_kwargs,\n"
        "            crucible_constraint=self.crucible_constraint,\n"
        "        )\n",
    ),
    (
        '            if finish_reason == "stop" and made_tool_call:\n'
        '                finish_reason = "tool_calls"\n'
        "\n",
        '            if finish_reason == "stop" and made_tool_call:\n'
        '                finish_reason = "tool_calls"\n'
        "\n"
        "            " + TAG + ":\n"
        "            # a matcher that failed stopped its answer; say so, not the text.\n"
        "            failed = args.crucible_constraint\n"
        "            if failed is not None and failed.error is not None:\n"
        "                from mlx_lm import _crucible_grammar\n"
        "\n"
        "                _crucible_grammar.answer_failure(self, failed)\n"
        "                return\n"
        "\n",
    ),
)

GENERATE_EDITS = (
    (
        "                for processor in self.logits_processors[e]:\n",
        "                " + TAG + ":\n"
        "                # extend() leaves None for a sequence with no processors.\n"
        "                for processor in self.logits_processors[e] or ():\n",
    ),
    (
        "        self.tokens = [self.tokens[idx] for idx in keep]\n"
        "        if any(self.samplers):\n"
        "            self.samplers = [self.samplers[idx] for idx in keep]\n"
        "        if any(self.logits_processors):\n"
        "            self.logits_processors = [self.logits_processors[idx] for idx in keep]\n"
        "        self.max_tokens = [self.max_tokens[idx] for idx in keep]\n",
        "        self.tokens = [self.tokens[idx] for idx in keep]\n"
        "        " + TAG + ":\n"
        "        # every per-sequence list keeps step with uids. Stock skips a list whose\n"
        "        # entries are all empty, which leaves it longer than uids, and the next\n"
        "        # sequence extended in then runs another row's processors.\n"
        "        if self.samplers:\n"
        "            self.samplers = [self.samplers[idx] for idx in keep]\n"
        "        if self.logits_processors:  # crucible: in step with uids\n"
        "            self.logits_processors = [self.logits_processors[idx] for idx in keep]\n"
        "        self.max_tokens = [self.max_tokens[idx] for idx in keep]\n",
    ),
)

MARKER = "_crucible_grammar.constraint_of_body(self.body)"
ABSENT_MARKER = "def _make_logits_processors(args):"

GENERATE_MARKER = "if self.logits_processors:  # crucible: in step with uids"
GENERATE_ABSENT_MARKER = "for processor in self.logits_processors[e]:"

FILES = (
    (REL, SERVER_EDITS, MARKER, ABSENT_MARKER),
    (GENERATE_REL, GENERATE_EDITS, GENERATE_MARKER, GENERATE_ABSENT_MARKER),
)


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


def patched_text(path: str, live: str, edits: tuple, marker: str, absent: str) -> str:
    for old, _ in edits:
        if live.count(old) != 1:
            print(
                f"ANCHOR_NOT_FOUND: expected exactly one {old.strip()!r} in {path}, "
                f"found {live.count(old)}",
                file=sys.stderr,
            )
            sys.exit(2)
    patched = live
    for old, new in edits:
        patched = patched.replace(old, new)
    if marker not in patched or absent in patched:
        print(f"INCOMPLETE: {path} still runs the stock code after the edits", file=sys.stderr)
        sys.exit(2)
    return patched


def main() -> None:
    prefix = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CRUCIBLE_LLM_ENV", "")
    ).rstrip("/")
    if not prefix:
        raise SystemExit(
            "usage: patch_mlx_lm_structured_output.py <env-prefix>   (or set CRUCIBLE_LLM_ENV)"
        )
    todo = []
    for rel, edits, marker, absent in FILES:
        path = site_packages_file(prefix, rel)
        with open(path, encoding="utf-8", newline="") as handle:
            live = handle.read()
        if marker in live and absent not in live:
            print("ALREADY_PATCHED " + path)
            continue
        todo.append((path, live, edits, marker, absent))
    if not todo:
        return

    check_version(prefix)
    # Every file's anchors are checked before any file is written: both or neither.
    planned = [
        (path, patched_text(path, live, edits, marker, absent))
        for path, live, edits, marker, absent in todo
    ]
    for path, patched in planned:
        shutil.copy2(path, path + ".orig")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(patched)
        print("PATCHED " + path)


if __name__ == "__main__":
    main()
