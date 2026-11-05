"""Destructive end-to-end for repair, on a throwaway CRUCIBLE_HOME and HF cache. Disk and network
only: nothing here loads a model or touches the card, and nothing real is in its path.

Builds a tiny real env with the real installer and pulls whisper-tiny and the RVC base (about 700
MB), damages and deletes a file of each, repairs, and verifies. Run it after changing
crucible/repair.py:

    python scripts/repair_e2e.py "$PWD"
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = sys.argv[1]
sys.path.insert(0, REPO)
sys.dont_write_bytecode = True
scratch = Path(tempfile.mkdtemp(prefix="repair-e2e-"))
home = scratch / "home"
os.environ["CRUCIBLE_HOME"] = str(home)
os.environ["HF_HOME"] = str(scratch / "hf")
for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
    os.environ.pop(name, None)
assert ".crucible" not in str(home)

from crucible import catalog, jobenv, repair  # noqa: E402
from crucible.backend import Backend, Gpu  # noqa: E402
from crucible.config import load_config, write_config  # noqa: E402
from crucible.narratorengines import declared_tts_footprints  # noqa: E402

BACKEND = Backend(kind="cuda-linux", platform="linux", arch="x86_64",
                  gpu=Gpu(vendor="nvidia", name="e2e", vram_bytes=24 * 1024**3), detail="e2e")
write_config(
    home, name="crucible@e2e", host="127.0.0.1", port=7199, token="e2e-not-a-secret",
    backend_kind="cuda-linux", enable_echo=True, enable_llm=False, enable_asr=False,
    enable_tts=False, enable_align=False, enable_rvc=False, enable_denoise=False,
    enable_image=False, enable_audio=False, enable_segment=False, enable_video=False,
    enable_score=False, retention_days=7, desktop_allowance_bytes=3 * 1024**3,
    desktop_allowance_basis="stated", desktop_allowance_note="",
    tts_engines=declared_tts_footprints("cuda-linux"),
)
config = load_config(home)

SPEC = jobenv.EnvSpec(job_type="e2e", key="e2e", recipe_name="e2e.txt", headline="six")
recipe = scratch / "e2e.txt"
recipe.write_text("six==1.16.0\nidna==3.7\n", encoding="utf-8")
jobenv.recipe_for = lambda spec: recipe
jobenv.every_env = lambda kind: (SPEC,)
jobenv.interpreter_for = lambda spec, h, kind, on_line=None: sys.executable
jobenv.SMOKE_IMPORT = {**jobenv.SMOKE_IMPORT, "e2e": {"cuda-linux": "six"}}
jobenv.refuse_without_room = lambda **kw: None

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def run(scope=None):
    events = []
    report = repair.run(config, BACKEND, scope or repair.Scope(), repair.NOBODY,
                        emit=lambda kind, data: events.append((kind, data)), on_line=lambda _l: None)
    return report, events


def corrupt(path):
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.unlink()  # a new inode: a hard link into the hub cache is damaged only here otherwise
    path.write_bytes(bytes(data))


try:
    # ── a real env, built by the real installer ──────────────────────────────────────────
    jobenv.install_env(home, SPEC, "cuda-linux", force=False, on_line=lambda _l: None)
    python = jobenv.env_python(home, SPEC)
    packages = next((jobenv.env_dir(home, SPEC) / "lib").glob("python*/site-packages"))
    catalog.subjects = lambda *a: []
    report, _ = run()
    check("a freshly built env reads healthy", report.healthy and not report.broken)

    six_py = packages / "six.py"
    good = six_py.read_bytes()
    six_py.write_bytes(good.replace(b"import", b"imqort", 1))
    report, _ = run(repair.Scope(verify_only=True))
    check("verify-only sees the damaged env and changes nothing",
          bool(report.broken) and six_py.read_bytes() != good, report.broken[0]["was"][:60] if report.broken else "")
    report, events = run()
    check("repair rebuilds the env", [r["subject"]["id"] for r in report.repaired] == ["e2e"])
    check("the damaged file is back byte for byte", six_py.read_bytes() == good)
    check("the rebuilt env imports", subprocess.run([str(python), "-c", "import six"]).returncode == 0)
    report, _ = run(repair.Scope(verify_only=True))
    check("it verifies healthy afterwards", report.healthy and not report.broken)

    (packages / "idna" / "core.py").unlink()
    report, _ = run()
    check("a deleted env file is rebuilt", report.healthy and bool(report.repaired)
          and (packages / "idna" / "core.py").is_file())

    # ── real weights: a snapshot subject and a pull_files subject ────────────────────────
    jobenv.every_env = lambda kind: ()
    import importlib
    importlib.reload(catalog)
    snapshot = catalog.find(config, BACKEND, "model", "whisper-tiny")
    base = catalog.find(config, BACKEND, "rvc-base", "base")
    assert snapshot is not None and base is not None
    for subject in (snapshot, base):
        subject.pull(force=False, on_line=lambda _l: None, on_progress=lambda *a: None)
        print(f"pulled {subject.kind} {subject.id}: {subject.installed().bytes / 1e6:.0f} MB", flush=True)

    for subject, label in ((snapshot, "snapshot"), (base, "pull_files")):
        catalog.subjects = lambda *a, s=subject: [s]
        found = subject.installed()
        report, _ = run()
        check(f"{label}: a fresh pull reads healthy", report.healthy and not report.broken and not report.unverified,
              f"unverified={report.unverified} ")
        pins = subject.pinned_files() if subject.pinned_files is not None else {}
        files = [p for p in repair._on_disk(found.path)
                 if not pins or p.relative_to(found.path).as_posix() in pins]
        victim = max(files, key=lambda p: p.stat().st_size)
        whole = victim.read_bytes()
        corrupt(victim)
        keeper = next((p for p in files if p != victim), None)
        keeper_inode = keeper.stat().st_ino if keeper is not None else None
        report, _ = run(repair.Scope(verify_only=True))
        check(f"{label}: verify-only sees {victim.name} damaged", bool(report.broken))
        report, _ = run()
        check(f"{label}: repair fetches it again", report.healthy and bool(report.repaired), str(report.failed)[:200])
        check(f"{label}: {victim.name} is back byte for byte", victim.read_bytes() == whole)
        if label == "snapshot" and keeper is not None:
            check(f"{label}: the good files were not fetched again", keeper.stat().st_ino == keeper_inode)
        report, _ = run(repair.Scope(verify_only=True))
        check(f"{label}: it verifies healthy afterwards", report.healthy and not report.broken)
        victim.unlink()
        report, _ = run()
        check(f"{label}: a deleted file comes back", report.healthy and victim.is_file() and victim.read_bytes() == whole,
              str(report.failed)[:200])
finally:
    shutil.rmtree(scratch, ignore_errors=True)
print(f"{sum(results)}/{len(results)} passed; throwaway home removed: {not scratch.exists()}")
sys.exit(0 if all(results) else 1)
