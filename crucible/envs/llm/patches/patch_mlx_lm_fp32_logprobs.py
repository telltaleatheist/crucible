import glob
import os
import re
import shutil
import sys

REL = "mlx_lm/generate.py"
VERSION_REL = "mlx_lm/_version.py"
EXPECTED_VERSION = "0.31.3"

TAG = "# PATCH (crucible 2026-09-24, envs/llm/patches/patch_mlx_lm_fp32_logprobs.py)"

HELPER_ANCHOR = "\n\ndef generate_step(\n"
HELPER = (
    "\n\n" + TAG + ":\n"
    "# the logprobs mlx-lm RETURNS are normalized in float32. Stock 0.31.3 takes\n"
    "# the log-sum-exp in the model's dtype (bf16), whose rounding shifts every\n"
    "# returned logprob by one common error of up to 0.0625, so a distribution's\n"
    "# mass read back came out 0.94-1.06. `sample` is the stock computation, and\n"
    "# the samplers still read it: what is generated does not change.\n"
    "def _crucible_logprobs(x, axis):\n"
    "    sample = x - mx.logsumexp(x, axis=axis, keepdims=True)\n"
    "    wide = x.astype(mx.float32)\n"
    "    return sample, wide - mx.logsumexp(wide, axis=axis, keepdims=True)\n"
    "\n\ndef generate_step(\n"
)

EDITS = (
    (
        "            logprobs = logits - mx.logsumexp(logits, keepdims=True)\n"
        "            sampled = sampler(logprobs)\n"
        "            return sampled, logprobs.squeeze(0)\n",
        "            " + TAG + "\n"
        "            logprobs, returned = _crucible_logprobs(logits, None)\n"
        "            sampled = sampler(logprobs)\n"
        "            return sampled, returned.squeeze(0)\n",
    ),
    (
        "        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)\n"
        "        y = sampler(logprobs)\n"
        "        return y, logprobs\n",
        "        " + TAG + "\n"
        "        logprobs, returned = _crucible_logprobs(logits, -1)\n"
        "        y = sampler(logprobs)\n"
        "        return y, returned\n",
    ),
    (
        "        # Normalize the logits\n"
        "        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)\n",
        "        # Normalize the logits\n"
        "        " + TAG + "\n"
        "        logprobs, returned = _crucible_logprobs(logits, -1)\n",
    ),
    (
        "        self._next_logprobs = list(logprobs)\n",
        "        self._next_logprobs = list(returned)\n",
    ),
)

MARKER = "wide = x.astype(mx.float32)"

ABSENT_MARKER = "logits - mx.logsumexp(logits"


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
            "usage: patch_mlx_lm_fp32_logprobs.py <env-prefix>   (or set CRUCIBLE_LLM_ENV)"
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
        print(f"INCOMPLETE: a stock site survived the edits in {path}", file=sys.stderr)
        sys.exit(2)

    shutil.copy2(path, path + ".orig")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(patched)
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
