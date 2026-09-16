"""Plan a patch release's packs and stage verified, unchanged inference packs.

This does not publish, upload, change versions, or operate an installed service.
Parts keep their original names: schema 1 resolves them beside the NEW manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import urllib.request

# Also work as `python scripts/release_packs.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crucible import envpack  # noqa: E402


def plan(source: envpack.PackManifest, version: str) -> dict:
    """Compare the source manifest with the checkout, never trusting pack names alone."""
    for value in (source.version, version):
        if not re.fullmatch(r"\d+\.\d+\.\d+", value):
            raise ValueError(f"expected a stable release version, got {value!r}")
    if tuple(map(int, version.split('.'))) <= tuple(map(int, source.version.split('.'))):
        raise ValueError("the target must be a new version after the source release")
    rows = []
    for name, backend in envpack.every_pack():
        target = envpack.pack_target(name, backend)
        blocker = None
        try:
            pin = envpack.python_for_target(target)
        except envpack.PackError as exc:
            blocker = str(exc)
            pin = None
        entry = source.find(name, backend)
        reason = None
        if target.job_type is None:
            reason = "runtime embeds Crucible source; always rebuild"
        elif entry is None:
            reason = "source release has no pack"
        elif entry.recipe_sha256 != envpack.recipe_digest(target.recipe):
            reason = "recipe changed"
        elif pin is None or entry.python != pin.python_version:
            reason = "standalone Python pin changed"
        rows.append({
            "name": name, "backend": backend,
            "action": "rebuild" if reason else "reuse",
            "reason": reason or "recipe and Python pin match",
            "build_blocker": blocker,
            "source_bytes": entry.bytes if entry else None,
        })
    return {"source_version": source.version, "version": version, "packs": rows}


def stage(source: envpack.PackManifest, version: str, name: str, backend: str, out: Path) -> Path:
    """Download one eligible pack, verify its joined digest/size, then write a fragment.

    `out` must not exist. A failed transfer leaves no usable manifest. No source
    release asset is changed, and there is no unverified upload path here.
    """
    row = next((p for p in plan(source, version)['packs']
                if p['name'] == name and p['backend'] == backend), None)
    if row is None or row['action'] != 'reuse':
        raise ValueError(f"{name}/{backend} is not reusable: {row}")
    if out.exists():
        raise ValueError(f"output already exists: {out}")
    entry = source.require(name, backend)
    expected_archive = envpack.pack_filename(name, backend, source.version)
    expected_parts = tuple(envpack.part_filename(expected_archive, i) for i in range(len(entry.parts)))
    if entry.parts != expected_parts:
        raise ValueError("source parts are not the ordered, release-scoped pack filenames")
    if not re.fullmatch(r"[a-f0-9]{64}", entry.sha256):
        raise ValueError("source pack has no SHA256 digest")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.release-pack-', dir=out.parent) as temporary:
        pending = Path(temporary)
        digest = hashlib.sha256()
        count = 0
        for part in entry.parts:
            url = envpack.asset_url(source.source, part)
            with urllib.request.urlopen(url, timeout=120) as response, (pending / part).open('wb') as dest:
                while data := response.read(1024 * 1024):
                    count += len(data)
                    if count > entry.bytes:
                        raise ValueError("source pack exceeds declared size")
                    digest.update(data)
                    dest.write(data)
        if count != entry.bytes or digest.hexdigest() != entry.sha256:
            raise ValueError("source pack size or SHA256 does not match its manifest")
        (pending / envpack.MANIFEST_NAME).write_text(
            envpack.PackManifest(version, (entry,)).dumps(), encoding='utf-8')
        # The destination is new and complete. Never merge into an old staging tree.
        pending.rename(out)
    return out / envpack.MANIFEST_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='source envpacks.json URL (file:// also works)')
    parser.add_argument('--version', required=True, help='planned new version; does not modify source')
    parser.add_argument('--check', action='store_true', help='exit nonzero when a required pack cannot be built')
    parser.add_argument('--stage', nargs=2, metavar=('PACK', 'BACKEND'), help='download one reusable pack')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if bool(args.stage) != bool(args.out):
        parser.error('--stage and --out must be used together')
    source = envpack.read_manifest(args.source)
    if args.stage:
        print(stage(source, args.version, *args.stage, args.out))
    else:
        result = plan(source, args.version)
        print(json.dumps(result, indent=2))
        if args.check and any(row['build_blocker'] for row in result['packs']):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
