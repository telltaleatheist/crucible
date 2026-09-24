"""Raise mlx-lm's `top_logprobs` ceiling from 11 to 40, so the Mac can read a
decision with more than seven options.

WHY. The decide door (PHASE22-DECIDE.md sections 2.4 and 2.6) asks the engine
for K = labels + 4 top logprobs and refuses `decide_not_served` when K would
pass the engine's stated cap. Stock mlx-lm 0.31.3 validates the chat body's
`top_logprobs` as `int, min 0, max 11, whitelist [-1]` and answers 400 above
it, so on the Mac a question with more than 7 options could not be read at all.
Briefcase needs 11 and 26. 40 covers 26 letters plus the door's margin of 4,
with room.

The validator is the ONLY ceiling. Read in the installed 0.31.3 `server.py` on
the Mac Studio on 2026-09-23: `_format_top_logprobs` takes whatever `top_n` it
is handed (`mx.argpartition(-logprobs, kth=top_n - 1)` over the whole
vocabulary), and both the batched and the single-stream generation paths pass
`args.top_logprobs` through untouched. Nothing downstream assumes 11.

USAGE: `<env python> patch_mlx_lm_top_logprobs.py <env prefix>` (or set
`CRUCIBLE_LLM_ENV`). The shape is `envs/tts/patches/patch_vllm.py`'s, on
purpose, so `crucible/narratorpatches.py` runs and checks both the same way:

- IDEMPOTENT. Already patched is asked of the LIVE file by the same marker
  `crucible doctor` greps for, and prints `ALREADY_PATCHED`.
- PATCHED FROM THE LIVE FILE, never from `.orig`. `.orig` is a snapshot of
  whatever was live just before this patch, kept for reference and never read
  back — patch_vllm.py's docstring records what reading it back once did
  (an upgrade's new source overwritten with the previous version's).
- REFUSES BY NAME. No file: `NOT_FOUND`. Two distinct site-packages trees:
  `AMBIGUOUS`. The anchor line not there: `ANCHOR_NOT_FOUND`, exit 2 — a newer
  mlx-lm that moved or reworded the validator must be re-patched deliberately,
  because a silently skipped patch is an engine that states 40 and answers 400.
"""
import glob
import os
import shutil
import sys

REL = "mlx_lm/server.py"

#: The line as mlx-lm 0.31.3 ships it (server.py L1245 on the Mac Studio,
#: 2026-09-23). Byte-exact, indentation included.
OLD = '        self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])'

NEW = (
    "        # PATCH (crucible 2026-09-23, envs/llm/patches/patch_mlx_lm_top_logprobs.py):\n"
    "        # the decide door reads up to 26 options plus a margin of 4; stock\n"
    "        # mlx-lm caps top_logprobs at 11. Nothing downstream assumes 11.\n"
    '        self._validate("top_logprobs", int, min_val=0, max_val=40, whitelist=[-1])'
)

#: What the doctor greps for: the EFFECTIVE line, so the check proves the
#: validator that runs rather than a comment beside it. Kept identical to
#: `crucible/envpatches.py`'s table; a keeper asserts this script writes it.
MARKER = 'self._validate("top_logprobs", int, min_val=0, max_val=40, whitelist=[-1])'

#: The stock line must be GONE, not merely joined by the new one.
ABSENT_MARKER = 'self._validate("top_logprobs", int, min_val=0, max_val=11, whitelist=[-1])'


def target_path() -> str:
    """`server.py` inside the env this was pointed at, deduped by real path.

    Globbed over `lib/python*/site-packages` because the env is built from
    whatever interpreter the server runs under. Deduped by REAL path for
    patch_vllm.py's reason (a `python3.1 -> python3.11` symlink matches twice);
    two DISTINCT trees are refused, because picking one is a coin flip nobody
    could see land.
    """
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
    # newline="" both ways: the file's own line endings are kept byte for byte.
    with open(path, encoding="utf-8", newline="") as handle:
        live = handle.read()

    if MARKER in live and ABSENT_MARKER not in live:
        print("ALREADY_PATCHED " + path)
        return

    if live.count(OLD) != 1:
        # Zero: upstream moved or reworded the validator. More than one: the
        # file is not the shape this patch was derived against. Either way the
        # patch must be re-derived by a person, not guessed at.
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
