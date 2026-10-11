#!/usr/bin/env python3
"""The `score` job against a RUNNING server on its card: does SheetSage2 transcribe a real song
into a YuE2-native score, and what does it cost?

  scripts/check-score-live.py --url http://127.0.0.1:7100 --token <token> song.mp3
  scripts/check-score-live.py --url ... song.mp3 other.flac --keep-harmony --max-seconds 120

GPU use: every song loads sheetsage2 (taking whatever is resident off the card, a YuE2
included) and runs SheetSage2 on it. Run it only with Owen's go for the card it uses. The
first run on a host also installs the score env (about 4 GB of wheels on cuda-linux) and pulls
the weights (2.8 GB) on submit; the script waits through `409 installing` and says so.

Nothing in the score job has run on a card before this (docs/SCORE.md "Unverified"). For each
song it prints how the job ended and, from the report: the windows and their tokens against
the 5,120 cap, SheetSage2's warnings, its peak GPU memory against the manifest's declared
figure, the seconds, the label decisions, the checks, and the score's bars and nominal
seconds. It passes (exit 0) when every song ends `done` with both scores written and its peak
under the declared figure; each one that does not says why. A failed song's report is fetched
from the job (`score.json` stays with a failed job) and its label record printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MODEL = "sheetsage2"
TERMINAL = {"done", "failed", "cancelled", "interrupted"}
MIB = 1024 * 1024


def call(args: argparse.Namespace, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json", "X-Crucible-Api": "1", "User-Agent": "check-score-live"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    request = urllib.request.Request(
        args.url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
            kind = response.headers.get("Content-Type", "")
            return response.status, json.loads(raw.decode("utf-8")) if "json" in kind else raw
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def submit(args: argparse.Namespace, song: Path) -> str:
    params: dict[str, Any] = {"keep_harmony": args.keep_harmony}
    if args.start_s:
        params["start_s"] = args.start_s
    if args.max_seconds is not None:
        params["max_seconds"] = args.max_seconds
    body = {
        "type": "score",
        "model": MODEL,
        "params": params,
        "inputs": {song.name: {"inline_base64": base64.b64encode(song.read_bytes()).decode("ascii")}},
    }
    deadline = time.monotonic() + args.install_timeout
    while True:
        status, answer = call(args, "POST", "/v1/jobs", body)
        if status == 202:
            return answer["job_id"]
        error = answer.get("error") or {}
        if status == 409 and error.get("code") == "installing" and time.monotonic() < deadline:
            print(f"  waiting: {error.get('message')}")
            time.sleep(30)
            continue
        raise SystemExit(f"{song.name}: refused {status} {error.get('code')}: {error.get('message')}")


def wait(args: argparse.Namespace, job_id: str) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        status, job = call(args, "GET", f"/v1/jobs/{job_id}")
        if status == 200 and job["status"] in TERMINAL:
            return job
        if time.monotonic() - started > args.timeout:
            raise SystemExit(f"job {job_id} did not end in {args.timeout} s")
        time.sleep(args.poll)


def report_of(args: argparse.Namespace, job_id: str, job: dict[str, Any]) -> dict[str, Any] | None:
    if job.get("score"):
        return job["score"]
    status, raw = call(args, "GET", f"/v1/jobs/{job_id}/artifacts/score.json")
    if status != 200:
        return None
    return raw if isinstance(raw, dict) else json.loads(raw)


def describe(song: Path, job_id: str, job: dict[str, Any], report: dict[str, Any] | None) -> list[str]:
    missed: list[str] = []
    print(f"{song.name}: job {job_id} {job['status']}")
    if job["status"] != "done":
        error = job.get("error") or {}
        missed.append(f"ended {job['status']} {error.get('code')}: {error.get('message')}")
    if report is None:
        missed.append("no report")
        return missed
    print(f"  {report['task']} on {report['device']} ({report['dtype']}), "
          f"{report['recording']['seconds']} s of {report['recording']['input']}")
    for window in report["windows"]:
        print(f"  window {window['window']}: {window['start_s']}-{window['end_s']} s, "
              f"{window['tokens']}/{window['token_cap']} tokens, {window['seconds']} s"
              + ("  <- AT ITS TOKEN LIMIT" if window["reached_token_limit"] else ""))
    for warning in report["warnings"]:
        print(f"  SheetSage2 warns: {warning}")
    peak = report.get("peak_gpu_mib")
    declared = report["memory_bytes_estimate"] / MIB
    print(f"  peak_gpu_mib {peak} (declared {declared:.0f} MiB), worker peak {report.get('peak_bytes')} B, "
          f"SheetSage2 {report.get('elapsed_seconds')} s, worker {report.get('seconds')} s")
    if report["device"] == "cuda" and peak is not None and peak > declared:
        missed.append(f"peak {peak:.0f} MiB above the declared {declared:.0f} MiB")
    native = report.get("yue2_native") or {}
    for record in native.get("labels") or []:
        kept = "kept" if record.get("kept") else f"dropped ({record.get('dropped_because')})"
        print(f"  label line {record['line']}: {record['sheetsage2']!r} -> {record['yue2']!r}, {kept}")
    for check in native.get("checks") or []:
        print(f"  check {check['check']}: {'passed' if check['passed'] else 'FAILED: ' + check.get('said', '')}")
    score = (report.get("lengths") or {}).get("score")
    if score:
        print(f"  score.abc: {score.get('bars')} bars, {score.get('seconds')} s at {score.get('bpm')} BPM, "
              f"{score.get('chars')} chars")
    if not report["artifacts"].get("score") or not report["artifacts"].get("instrumental"):
        missed.append("score.abc and instrumental.abc were not both written")
    return missed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("songs", nargs="+", type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", default=None, help="omit for a server with [auth] mode = none")
    parser.add_argument("--keep-harmony", action="store_true")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds a job may run")
    parser.add_argument("--install-timeout", type=float, default=3600.0,
                        help="seconds to wait through an install on submit")
    parser.add_argument("--poll", type=float, default=2.0)
    args = parser.parse_args()
    failed = 0
    for song in args.songs:
        job_id = submit(args, song)
        job = wait(args, job_id)
        missed = describe(song, job_id, job, report_of(args, job_id, job))
        for why in missed:
            print(f"  MISS {why}")
        failed += bool(missed)
    print(f"{len(args.songs) - failed} of {len(args.songs)} song(s) passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
