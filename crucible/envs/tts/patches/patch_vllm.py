"""Patch vLLM 0.28's negative-token-id rejection so vllm-omni's higgs_audio_v3
voice-clone path works.

PROVENANCE. Copied from
    E:\\training\\_campaigns\\2026-09-01-cod-full-rebuild\\higgs\\work\\patch_vllm.py
(owens-pc, 2026-09-04), the script that made voice cloning work in the `higgs3`
env every measurement in electron/data/higgs-models.json was taken against. The
OLD/NEW anchors and the replacement are byte-identical to that original; the only
change is that the target path is resolved from the env prefix (argv[1] or
$HIGGS_ENV) instead of being hardcoded to /home/telltale, so this ships. Nothing
at run time reads E:\\training.

IDEMPOTENT: it always patches from the pristine `.orig` copy, so running it twice
is the same as running it once, and running it after a pip upgrade re-applies it
(which is required — an upgrade replaces the file and silently reverts this).

vllm-omni builds the clone prompt as
    <|tts|> [<|ref_text|> ...] <|ref_audio|> [-100]*N <|text|> ... <|audio|>
and substitutes the -100 positions with fused reference-audio codebook embeddings
at prefill (higgs_audio_v3_talker._replace_placeholders). vLLM 0.28 added a blanket
`min_input_id < 0 -> out of vocabulary` check in v1/engine/input_processor.py which
rejects the request before it ever reaches the talker. Allow exactly -100
(vllm_omni...higgs_audio_v3_tokenizer.AUDIO_PLACEHOLDER_ID).
"""
import glob
import os
import shutil
import sys

REL = "vllm/v1/engine/input_processor.py"
#: What the doctor greps for. Kept identical to HIGGS_PATCHES in
#: electron/tool-paths.ts; a keeper asserts this script writes it.
MARKER = "min_input_id != -100"


def target_path() -> str:
    """The file to patch, inside the env this was pointed at.

    The python version is GLOBBED rather than assumed: the original hardcoded
    `python3.11`, which is right for the env BookForge's installer builds and
    wrong the moment someone points at an env they built with another version.

    DEDUPED BY REAL PATH, which is not a nicety — conda ships a
    `lib/python3.1 -> python3.11` symlink, so the glob matches the same file
    twice on a perfectly normal env and a naive count would refuse it. What the
    refusal below is actually for is two DISTINCT site-packages trees under one
    prefix, i.e. the caller pointed at something that is not a conda env; picking
    one of those would be a coin flip nobody could see land.
    """
    prefix = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HIGGS_ENV", "")).rstrip("/")
    if not prefix:
        raise SystemExit("usage: patch_vllm.py <conda-env-prefix>   (or set HIGGS_ENV)")
    hits = sorted({os.path.realpath(p) for p in glob.glob(f"{prefix}/lib/python*/site-packages/{REL}")})
    if not hits:
        raise SystemExit(f"NOT_FOUND: no {REL} under {prefix}/lib/python*/site-packages")
    if len(hits) > 1:
        raise SystemExit(f"AMBIGUOUS: {len(hits)} distinct site-packages trees under {prefix}: {hits}")
    return hits[0]

OLD = """            if min_input_id < 0:
                raise VLLMValidationError(
                    f"Token id {min_input_id} is out of vocabulary"
                )"""

NEW = """            # PATCH (bookforge 2026-09-04): vllm-omni's higgs_audio_v3 voice-clone
            # path deliberately emits AUDIO_PLACEHOLDER_ID == -100 into
            # prompt_token_ids; the talker replaces those positions with fused
            # reference-audio embeddings at prefill. Let exactly -100 through.
            if min_input_id < 0 and min_input_id != -100:
                raise VLLMValidationError(
                    f"Token id {min_input_id} is out of vocabulary"
                )"""


def main():
    P = target_path()
    live = open(P).read()

    # ALREADY PATCHED? Ask the LIVE file, by the same marker the doctor greps
    # for. Idempotent, and it is what makes re-running the installer safe.
    if MARKER in live:
        print("ALREADY_PATCHED " + P)
        return

    # PATCH FROM THE LIVE FILE, never from `.orig`.
    #
    # This used to read `.orig`, which was written once and never refreshed —
    # so after a pip UPGRADE in the env, `.orig` held the PREVIOUS version's
    # source, and re-running the installer wrote that old content back over the
    # new site-packages file. The doctor's marker grep then certified stale code
    # as patched, all-green, silently. `.orig` is now a snapshot of whatever was
    # live just before this patch, kept for reference and never read back.
    if OLD not in live:
        # Not "already patched" — the marker check above ruled that out. A missing
        # anchor means upstream changed this code and the patch must be re-derived
        # against the new version. Failing loud is the point: a silently-skipped
        # patch is an engine that looks installed and does not work.
        print("ANCHOR_NOT_FOUND", file=sys.stderr)
        sys.exit(2)

    shutil.copy2(P, P + ".orig")
    open(P, "w").write(live.replace(OLD, NEW))
    print("PATCHED " + P)


if __name__ == "__main__":
    main()
