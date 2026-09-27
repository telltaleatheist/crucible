#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = """Chapter Four

The morning came in grey and the harbour was still. Nobody had
told them what to expect, and so they waited — as people do — for
someone with more authority to arrive and explain the situa-
tion. By noon the tide had turned twice and the explanation had
not come. "We could go in ourselves," said the youngest of them,
and the others looked at the water and said nothing at all."""

PROMPT = (
    "Clean this passage for text-to-speech narration. Join words broken "
    "across line breaks, join lines that are one paragraph, and change "
    "nothing else. Output only the cleaned text."
)


ALREADY_CLEAR = "model_not_resident"


def error_code(detail: str) -> str | None:
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


class Refused(Exception):

    def __init__(self, what: str, detail: str, *, status: int | None = None) -> None:
        self.detail = detail
        self.status = status
        self.code = error_code(detail)
        where = "" if status is None else f"HTTP {status} "
        super().__init__(f"{what}: {where}{detail[:600]}")


def post(url: str, token: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Crucible-Api": "1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise Refused(
            f"{url} refused",
            exc.read().decode("utf-8", "replace"),
            status=exc.code,
        ) from None
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"{url} did not answer: {type(exc).__name__}: {exc}")


def run_job(base: str, token: str, request: dict, timeout: float) -> dict:
    job = post(f"{base}/v1/jobs", token, request, timeout)
    job_id = job.get("job_id") or job.get("id")
    if job_id is None:
        raise SystemExit(f"{request['type']} was accepted without an id: {job}")
    while True:
        poll = urllib.request.Request(
            f"{base}/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {token}", "X-Crucible-Api": "1"},
        )
        with urllib.request.urlopen(poll, timeout=60) as response:
            status = json.loads(response.read().decode("utf-8"))
        if status["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(2.0)
    if status["status"] != "done":
        raise Refused(
            f"{request['type']} {request.get('model', '')} {status['status']}",
            json.dumps(status),
        )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one cleanup chunk through a Crucible server and time it.",
    )
    parser.add_argument("--server", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = args.server.rstrip("/")

    started = time.monotonic()
    try:
        run_job(
            base, args.token, {"type": "load-model", "model": args.model}, args.timeout
        )
    except Refused as exc:
        raise SystemExit(str(exc)) from None
    load_seconds = time.monotonic() - started

    cleaning: SystemExit | Refused | None = None
    try:
        clean_one_chunk(base, args.token, args.model, out, load_seconds, args.timeout)
    except (SystemExit, Refused) as exc:
        cleaning = exc
        print(f"THE CLEANUP FAILED: {exc}", file=sys.stderr)

    tidying: Exception | None = None
    print(f"unload-model {args.model}")
    try:
        run_job(
            base,
            args.token,
            {"type": "unload-model", "model": args.model},
            args.timeout,
        )
        print("  unloaded")
    except Refused as exc:
        if exc.code == ALREADY_CLEAR:
            print(f"  the card was already clear of {args.model}")
        else:
            tidying = exc
            print(f"THE UNLOAD FAILED: {exc}", file=sys.stderr)
    except SystemExit as exc:
        tidying = RuntimeError(str(exc))
        print(f"THE UNLOAD FAILED: {exc}", file=sys.stderr)

    if cleaning is not None:
        return 1
    return 1 if tidying is not None else 0


def clean_one_chunk(
    base: str, token: str, model: str, out: Path, load_seconds: float, timeout: float
) -> None:
    started = time.monotonic()
    answer = post(
        f"{base}/v1/openai/chat/completions",
        token,
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": CHUNK},
            ],
        },
        timeout,
    )
    seconds = time.monotonic() - started
    (out / "cleanup.json").write_text(json.dumps(answer, indent=2), encoding="utf-8")

    content = answer["choices"][0]["message"].get("content") or ""
    if content.strip() == "":
        raise SystemExit(
            "the model answered with no content at all. That is the "
            "reasoning-budget failure `[defaults] thinking = false` exists "
            f"for; the raw answer is in {out / 'cleanup.json'}"
        )
    joined = "situation" in content and "situa-" not in content
    print(f"server:          {base}")
    print(f"model:           {model}")
    print(f"seconds to load: {load_seconds:.1f}")
    print(f"seconds/chunk:   {seconds:.1f}")
    print(f"joined the broken word: {joined}")
    print(f"artifacts:       {out}")


if __name__ == "__main__":
    raise SystemExit(main())
