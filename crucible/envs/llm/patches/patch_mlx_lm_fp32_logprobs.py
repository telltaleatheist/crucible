"""Make the logprobs mlx-lm RETURNS float32, so a decision's label mass is a
probability mass.

WHY. mlx-lm 0.31.3 normalizes with `logprobs = logits - mx.logsumexp(logits)`
in the model's own dtype, which for Crucible's Qwen3.5/3.8 builds is bfloat16
(`mlx_lm/generate.py` L420, L549, L1352 — every site whose result is returned,
read on the Mac Studio 2026-09-24). bfloat16 keeps 8 significant bits: at a
log-sum-exp between 16 and 32 its spacing is 0.125, so the rounded lse is off by
up to 0.0625 and EVERY returned logprob is shifted by that one error. The mass
of a distribution read back through `exp` is then off by a common factor of up
to exp(+-0.0625) = 0.939-1.065. That is what the decide door measured: a
qwen3.5-2b triage over 1,362 yes/no questions had `label_mass` quantiles 5%
0.940, median 0.997, 75% 1.023, 95% 1.055 — above 1, which no probability mass
can be. Reproduced on the Mac's CPU with mlx itself (2026-09-24, 400 synthetic
peaked distributions at logit ~25 over the 248320 vocabulary): bf16 mass
0.946 / 1.008 / 1.056 (5% / median / 95%), float32 0.998 / 1.000 / 1.000.

WHAT CHANGES, AND WHAT DOES NOT. Each site now computes BOTH: the stock-dtype
logprobs, which the sampler reads exactly as before — so what is GENERATED is
unchanged at every temperature, not only at 0 — and a float32 copy
(`logits.astype(mx.float32)` before `logsumexp`), which is what the site
returns and therefore what `server.py` reports as `logprob` and
`top_logprobs`. The cost is one extra log-sum-exp over the vocabulary per
sequence per step, about 1 MB of float32 per row.

USAGE: `<env python> patch_mlx_lm_fp32_logprobs.py <env prefix>` (or set
`CRUCIBLE_LLM_ENV`), the same shape as `patch_mlx_lm_top_logprobs.py`:

- IDEMPOTENT: already patched is asked of the LIVE file by the marker `crucible
  doctor` greps for, and prints `ALREADY_PATCHED`.
- PATCHED FROM THE LIVE FILE, never from `.orig`.
- VERSION-PINNED: derived against mlx-lm 0.31.3 byte for byte, and it refuses
  `VERSION_MISMATCH` (exit 2) on any other `mlx_lm/_version.py`, before it
  looks at an anchor.
- ALL OR NOTHING: every anchor must be there exactly once or nothing is
  written (`ANCHOR_NOT_FOUND`, exit 2). A file with some sites float32 and
  some not would be an engine that states one precision and returns two.
"""
import glob
import os
import re
import shutil
import sys

REL = "mlx_lm/generate.py"
VERSION_REL = "mlx_lm/_version.py"
EXPECTED_VERSION = "0.31.3"

TAG = "# PATCH (crucible 2026-09-24, envs/llm/patches/patch_mlx_lm_fp32_logprobs.py)"

#: The helper, inserted once before `generate_step`. Its own lines never spell
#: `logits - mx.logsumexp(logits`, so `ABSENT_MARKER` can prove that no stock
#: site is left anywhere in the file.
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

#: (stock, patched) pairs, byte-exact, indentation included. `generate_step`'s
#: `_step` (L420-422), `speculative_generate_step`'s `_process_and_sample`
#: (L549-551), and `GenerationBatch._step` (L1351-1352 and L1368) — the
#: batched path the server takes for every batchable request.
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

#: What the doctor greps for: the EFFECTIVE float32 line of the helper.
#: Kept identical to `crucible/envpatches.py`'s table; a keeper asserts it.
MARKER = "wide = x.astype(mx.float32)"

#: The stock normalization must be GONE from every site, not merely joined by
#: the helper.
ABSENT_MARKER = "logits - mx.logsumexp(logits"


def site_packages_file(prefix: str, rel: str) -> str:
    """`rel` inside the env, deduped by real path (patch_vllm.py's reason)."""
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
        # Unreachable on the file this was derived against; kept so a future
        # edit to EDITS that misses a site cannot write a half-patched file.
        print(f"INCOMPLETE: a stock site survived the edits in {path}", file=sys.stderr)
        sys.exit(2)

    shutil.copy2(path, path + ".orig")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(patched)
    print("PATCHED " + path)


if __name__ == "__main__":
    main()
