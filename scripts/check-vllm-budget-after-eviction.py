#!/usr/bin/env python3
"""A vLLM load that follows an eviction, N times, through a RUNNING server's job API: what
free memory each load's plan rested on, the utilization it gave vLLM, and the free memory
vLLM itself saw, against the card's idle free memory.

  scripts/check-vllm-budget-after-eviction.py --url http://127.0.0.1:7100 --token <token>
  scripts/check-vllm-budget-after-eviction.py --url ... --token ... --first qwen3.5-0.8b \\
      --second qwen3.5-4b-8bit --count 10
  scripts/check-vllm-budget-after-eviction.py --url ... --token ... --second qwen3.5-4b-8bit \\
      --first-body '{"type": "image", "model": "dreamshaper-xl-lightning", "params": {"prompt": "a lighthouse"}}'

GPU use: it loads two models on the server's card, alternately, --count times each. Run it
only with Owen's go for the card it uses (Victoria's 8 GiB laptop is the one it is for).

Why (2026-10-11): on Victoria's RTX 3070 Laptop (Crucible 1.0.140) qwen3.5-4b-8bit loaded
with nothing resident and failed when it followed yue2-3b: "Free memory on device cuda:0
(6.95/8.0 GiB) on startup is less than desired GPU memory utilization (0.875, 7.0 GiB)".
The plan had been made BEFORE the eviction, from the card's free memory plus the
resident's reclaimable figure (its estimate, not what it held), which came to more than an
idle 8 GiB card has. The fix plans from the card read after the eviction, once its free
memory stops rising (crucible/jobs/llm `plan_on_the_card`). This is that fix on the card.

What it does, --count times:

  1. Makes --first resident: a load-model job for it, or the job --first-body states (an
     `image` or `audio` job leaves its model resident, which is the incident's shape).
  2. Reads GET /v1/accelerator: the card's free memory with the first model resident.
  3. Loads --second (a vLLM model) with a load-model job, which evicts the first. From the
     job's events it reads the plan line the load says once it has read the card ("KV pool
     ... of a B GiB budget (F GiB free, ...)"): F is the free memory the plan rested on and
     B / total the --gpu-memory-utilization vLLM was given. A load that failed is read for
     vLLM's own "Free memory on device ... (S/T GiB) on startup" sentence: S is what vLLM saw.
     With --read-logs, run ON the server's machine, the engine log the load names is read for
     the same sentence on a load that succeeded (vLLM does not always print it then).

Then it unloads --second, waits --idle-wait seconds, and reads the card's idle free memory.

A row passes when the load is `done` and its plan's free memory is no more than the idle
card's (within --tolerance-mib, the desktop's own drift between readings): a plan cannot
rest on memory the card does not have. The run passes (exit 0) when every row does. Before
the fix the incident's plans read about 8 GiB free on a card whose idle free is 7.83 GiB.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TERMINAL = {"done", "failed", "cancelled", "interrupted"}
GIB = 2**30
MIB = 2**20

PLAN_LINE = re.compile(
    r"KV pool (?P<pool>[\d.]+) GiB of a (?P<budget>[\d.]+) GiB budget "
    r"\((?P<free>[\d.]+) GiB free"
)
VLLM_SAW = re.compile(
    r"Free memory on device \S+ \((?P<free>[\d.]+)/(?P<total>[\d.]+) GiB\) on startup"
    r"(?: is less than desired GPU memory utilization \((?P<util>[\d.]+), "
    r"(?P<want>[\d.]+) GiB\))?"
)
LOG_PATH = re.compile(r"; log (?P<path>\S+)$")


def gib(value: float | int | None) -> str:
    return "-" if value is None else f"{value / GIB:.2f}"


class Server:
    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> urllib.request.Request:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Crucible-Api": "1",
            "User-Agent": "check-vllm-budget-after-eviction",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode("utf-8"),
            headers=headers,
            method=method,
        )

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
        try:
            with urllib.request.urlopen(self._request(method, path, body), timeout=120) as response:
                return response.status, json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8") or "{}")

    def events(self, job_id: str) -> list[dict[str, Any]]:
        """The job's whole event stream (it ends once the job is terminal)."""
        events: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        request = self._request("GET", f"/v1/jobs/{job_id}/events", None)
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line == "":
                    if current:
                        events.append(current)
                        current = {}
                    continue
                field, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if field == "event":
                    current["event"] = value
                elif field == "data":
                    current["data"] = json.loads(value)
        if current:
            events.append(current)
        return events

    def card(self) -> dict[str, Any]:
        status, body = self.call("GET", "/v1/accelerator")
        if status != 200:
            raise SystemExit(f"GET /v1/accelerator answered {status}: {body}")
        return body

    def run(self, body: dict[str, Any], timeout: float, poll: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        status, submitted = self.call("POST", "/v1/jobs", body)
        if status != 202:
            return {"status": f"refused {status}", "error": submitted.get("error")}, []
        job_id = submitted["job_id"]
        started = time.monotonic()
        while True:
            status, job = self.call("GET", f"/v1/jobs/{job_id}")
            if status == 200 and job["status"] in TERMINAL:
                break
            if time.monotonic() - started > timeout:
                return {"id": job_id, "status": "timed out"}, []
            time.sleep(poll)
        return job, self.events(job_id)


def messages(events: list[dict[str, Any]]) -> list[str]:
    said = []
    for event in events:
        data = event.get("data") or {}
        if isinstance(data.get("message"), str):
            said.append(data["message"])
        error = data.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            said.append(error["message"])
    return said


def vllm_saw(text: str) -> dict[str, float] | None:
    found = VLLM_SAW.search(text)
    if found is None:
        return None
    return {key: float(value) for key, value in found.groupdict().items() if value is not None}


def second_load(server: Server, args: argparse.Namespace, total: int) -> dict[str, Any]:
    body: dict[str, Any] = {"type": "load-model", "model": args.second}
    if args.context is not None:
        body["params"] = {"context": args.context}
    began = time.monotonic()
    job, events = server.run(body, args.timeout, args.poll)
    said = messages(events)
    row: dict[str, Any] = {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "seconds": round(time.monotonic() - began, 1),
        "plan_free": None,
        "budget": None,
        "utilization": None,
        "vllm_saw": None,
        "settle_note": None,
        "error": (job.get("error") or {}).get("message") if isinstance(job.get("error"), dict) else job.get("error"),
    }
    for line in said:
        plan = PLAN_LINE.search(line)
        if plan is not None:
            row["plan_free"] = float(plan["free"]) * GIB
            row["budget"] = float(plan["budget"]) * GIB
            row["utilization"] = row["budget"] / total
        if "still rising" in line:
            row["settle_note"] = line
        saw = vllm_saw(line)
        if saw is not None:
            row["vllm_saw"] = saw
    if row["vllm_saw"] is None and args.read_logs:
        path = next((m["path"] for m in (LOG_PATH.search(line) for line in said) if m), None)
        if path is not None and Path(path).is_file():
            row["vllm_saw"] = vllm_saw(Path(path).read_text(encoding="utf-8", errors="replace"))
            row["log"] = path
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--first", default="qwen3.5-0.8b", help="the model made resident first (load-model)")
    parser.add_argument("--first-body", help="a job body (JSON) that makes the first model resident instead")
    parser.add_argument("--second", default="qwen3.5-4b-8bit", help="the vLLM model whose load evicts the first")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--context", type=int)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--idle-wait", type=float, default=10.0)
    parser.add_argument("--tolerance-mib", type=int, default=128)
    parser.add_argument("--read-logs", action="store_true", help="read the engine log each load names (run on the server)")
    parser.add_argument("--json", help="also write every row to this file")
    args = parser.parse_args()

    server = Server(args.url, args.token)
    first_body = json.loads(args.first_body) if args.first_body else {"type": "load-model", "model": args.first}
    first_name = first_body.get("model") or "the first job's model"
    if first_name == args.second:
        raise SystemExit("--first and --second are the same model, so nothing would be evicted")
    total = server.card()["gpu"]["total_bytes"]
    print(f"card total {gib(total)} GiB; {args.count} x ({first_name} resident, then {args.second})\n")

    rows: list[dict[str, Any]] = []
    for index in range(args.count):
        job, _ = server.run(first_body, args.timeout, args.poll)
        if job.get("status") != "done":
            print(f"#{index}: making {first_name} resident ended {job.get('status')}: {job.get('error')}")
            rows.append({"index": index, "status": "first-not-resident", "first": job.get("status")})
            continue
        before = server.card()
        row = {"index": index, "free_with_first": before["free_bytes"], "resident_before": before.get("resident")}
        row.update(second_load(server, args, total))
        rows.append(row)
        saw = row["vllm_saw"] or {}
        utilization = "-" if row["utilization"] is None else f"{row['utilization']:.4f}"
        print(
            f"#{index}: {row['status']:<8} {row['seconds']:>6}s  free with first {gib(row['free_with_first'])}  "
            f"plan free {gib(row['plan_free'])}  budget {gib(row['budget'])}  util {utilization}  "
            f"vLLM saw {saw.get('free', '-')} GiB"
            + (f"  ERROR {row['error']}" if row["status"] != "done" else "")
            + (f"  ({row['settle_note']})" if row["settle_note"] else "")
        )

    server.run({"type": "unload-model", "model": args.second}, args.timeout, args.poll)
    time.sleep(args.idle_wait)
    idle = server.card()
    idle_free = idle["free_bytes"]
    tolerance = args.tolerance_mib * MIB
    print(f"\nidle card after unloading {args.second}: {gib(idle_free)} GiB free (resident: {idle.get('resident')})")

    misses = []
    for row in rows:
        if row.get("status") != "done":
            misses.append(f"#{row['index']}: ended {row.get('status')}: {row.get('error')}")
            continue
        if row.get("plan_free") is None:
            misses.append(f"#{row['index']}: no plan line in the load's events")
            continue
        if row["plan_free"] > idle_free + tolerance:
            misses.append(
                f"#{row['index']}: the plan rested on {gib(row['plan_free'])} GiB free, more than the idle "
                f"card's {gib(idle_free)} GiB"
            )
    if args.json:
        Path(args.json).write_text(json.dumps({"idle_free": idle_free, "total": total, "rows": rows}, indent=2), encoding="utf-8")
    print("\nPASS" if not misses else "\nMISSES:\n  " + "\n  ".join(misses))
    return 0 if not misses else 1


if __name__ == "__main__":
    sys.exit(main())
