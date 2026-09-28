import filecmp
import glob
import os
import shutil
import sys

REL = "mlx_lm/_crucible_items.py"
PACKAGE_REL = "mlx_lm/server.py"

SOURCE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "engines", "items_forward.py"
)


def package_dir(prefix: str) -> str:
    hits = sorted(
        {
            os.path.realpath(os.path.dirname(p))
            for p in glob.glob(f"{prefix}/lib/python*/site-packages/{PACKAGE_REL}")
        }
    )
    if not hits:
        raise SystemExit(f"NOT_FOUND: no {PACKAGE_REL} under {prefix}/lib/python*/site-packages")
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
            "usage: patch_mlx_lm_decide_items_helper.py <env-prefix>   (or set CRUCIBLE_LLM_ENV)"
        )
    if not os.path.isfile(SOURCE):
        print(f"NO_SOURCE: {SOURCE} is not there", file=sys.stderr)
        sys.exit(2)
    target = os.path.join(package_dir(prefix), os.path.basename(REL))
    if os.path.isfile(target) and filecmp.cmp(SOURCE, target, shallow=False):
        print("ALREADY_PATCHED " + target)
        return
    shutil.copyfile(SOURCE, target)
    print("PATCHED " + target)


if __name__ == "__main__":
    main()
