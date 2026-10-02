import glob
import os
import re
import shutil
import sys

REL = "mlx_lm/tokenizer_utils.py"
VERSION_REL = "mlx_lm/_version.py"
EXPECTED_VERSION = "0.31.3"

TAG = "# PATCH (crucible 2026-10-01, envs/llm/patches/patch_mlx_lm_detokenizer_tokenmap.py)"

HELPER_ANCHOR = "\n\nclass StreamingDetokenizer:\n"
HELPER = (
    "\n\n" + TAG + ":\n"
    "# a streaming detokenizer's id-to-token table is built once per tokenizer and\n"
    "# shared. Stock 0.31.3 rebuilds it for EVERY request from tokenizer.vocab, which\n"
    "# a fast tokenizer materializes as a fresh dict on each read: ~150 ms on\n"
    "# Qwen3.5's 248k vocabulary (measured on the Mac Studio, 2026-10-01), on the one\n"
    "# generation thread, before the request's first forward. The table is only\n"
    "# ever read, so one copy serves every request.\n"
    "def _crucible_tokenmap(tokenizer, attr, build):\n"
    "    tokenmap = getattr(tokenizer, attr, None)\n"
    "    if tokenmap is None:\n"
    "        tokenmap = build(tokenizer.vocab)\n"
    "        setattr(tokenizer, attr, tokenmap)\n"
    "    return tokenmap\n"
    "\n"
    "\n"
    "def _crucible_bpe_tokenmap(vocab):\n"
    "    tokenmap = [None] * len(vocab)\n"
    "    for value, tokenid in vocab.items():\n"
    "        tokenmap[tokenid] = value\n"
    "    return tokenmap\n"
    "\n"
    "\n"
    "def _crucible_spm_tokenmap(vocab):\n"
    "    tokenmap = [\"\"] * (max(vocab.values()) + 1)\n"
    "    for value, tokenid in vocab.items():\n"
    "        if value.startswith(\"<0x\"):\n"
    "            # Replace bytes with their value\n"
    "            tokenmap[tokenid] = bytes([int(value[3:5], 16)])\n"
    "        else:\n"
    "            tokenmap[tokenid] = value.encode()\n"
    "    return tokenmap\n"
    "\n\nclass StreamingDetokenizer:\n"
)

EDITS = (
    (
        "        # Extract the tokens in a list from id to text\n"
        "        self.tokenmap = [\"\"] * (max(tokenizer.vocab.values()) + 1)\n"
        "        for value, tokenid in tokenizer.vocab.items():\n"
        "            if value.startswith(\"<0x\"):\n"
        "                # Replace bytes with their value\n"
        "                self.tokenmap[tokenid] = bytes([int(value[3:5], 16)])\n"
        "            else:\n"
        "                self.tokenmap[tokenid] = value.encode()\n",
        "        # Extract the tokens in a list from id to text\n"
        "        " + TAG + "\n"
        "        self.tokenmap = _crucible_tokenmap(\n"
        "            tokenizer, \"_crucible_spm_tokenmap\", _crucible_spm_tokenmap\n"
        "        )\n",
    ),
    (
        "        # Extract the tokens in a list from id to text\n"
        "        self.tokenmap = [None] * len(tokenizer.vocab)\n"
        "        for value, tokenid in tokenizer.vocab.items():\n"
        "            self.tokenmap[tokenid] = value\n",
        "        # Extract the tokens in a list from id to text\n"
        "        " + TAG + "\n"
        "        self.tokenmap = _crucible_tokenmap(\n"
        "            tokenizer, \"_crucible_bpe_tokenmap\", _crucible_bpe_tokenmap\n"
        "        )\n",
    ),
)

MARKER = 'tokenizer, "_crucible_bpe_tokenmap", _crucible_bpe_tokenmap'

ABSENT_MARKER = "for value, tokenid in tokenizer.vocab.items():"


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
            "usage: patch_mlx_lm_detokenizer_tokenmap.py <env-prefix>   "
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
        print(f"INCOMPLETE: a per-request table build survived in {path}", file=sys.stderr)
        sys.exit(2)

    shutil.copy2(path, path + ".orig")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(patched)
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
