import glob
import os
import shutil
import sys

REL = "mlx_lm/models/qwen3.py"

OLD = (
    "    def sanitize(self, weights):\n"
    "        if self.args.tie_word_embeddings:\n"
    '            weights.pop("lm_head.weight", None)\n'
    "        return weights\n"
)

NEW = (
    "    def sanitize(self, weights):\n"
    "        # PATCH (crucible 2026-10-10, envs/llm/patches/patch_mlx_lm_qwen3_bare_checkpoint.py):\n"
    "        # Qwen3-Embedding is saved from transformers' AutoModel: the bare Qwen3Model,\n"
    '        # its keys without the "model." prefix and no lm_head. Read as this Model,\n'
    "        # the keys are prefixed and an untied head is given the input embeddings:\n"
    "        # an embedding is read off the hidden state, never through the head, and\n"
    "        # the engine's start check generates its one token through it.\n"
    '        if "embed_tokens.weight" in weights and not any(\n'
    '            key.startswith("model.") for key in weights\n'
    "        ):\n"
    '            weights = {"model." + key: value for key, value in weights.items()}\n'
    "            if not self.args.tie_word_embeddings:\n"
    '                weights["lm_head.weight"] = weights["model.embed_tokens.weight"]\n'
    "        if self.args.tie_word_embeddings:\n"
    '            weights.pop("lm_head.weight", None)\n'
    "        return weights\n"
)

MARKER = 'weights = {"model." + key: value for key, value in weights.items()}'


def target_path() -> str:
    prefix = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CRUCIBLE_LLM_ENV", "")
    ).rstrip("/")
    if not prefix:
        raise SystemExit(
            "usage: patch_mlx_lm_qwen3_bare_checkpoint.py <env-prefix>   "
            "(or set CRUCIBLE_LLM_ENV)"
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

    if MARKER in live:
        print("ALREADY_PATCHED " + path)
        return

    if live.count(OLD) != 1:
        print(
            f"ANCHOR_NOT_FOUND: expected exactly one Model.sanitize as mlx-lm 0.31.3 "
            f"writes it in {path}, found {live.count(OLD)}",
            file=sys.stderr,
        )
        sys.exit(2)

    shutil.copy2(path, path + ".orig")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(live.replace(OLD, NEW))
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
