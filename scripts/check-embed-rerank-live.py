#!/usr/bin/env python3
"""The embed and rerank verbs against a RUNNING server, on its card: what only a card can
prove. Needs the retrieval package installed there (`crucible install retrieval`) and the
card free for a load; it loads each model by calling the verb (the server's line loads it).

  scripts/check-embed-rerank-live.py --url http://127.0.0.1:7100 --token <token>

GPU use: it loads Qwen3-Embedding-8B and Qwen3-Reranker-8B (about 16 GB each) and, with
--general, the decide model named. Run it only with Owen's go for the card it uses.

What it proves, each a numbered check that passes or fails by itself:
 1. embed reproduces the model card's own numbers: the cosine matrix of its two queries
    (input_type query, the default instruction) against its two documents is within 0.01
    of [[0.7493, 0.0751], [0.0880, 0.6318]] (the card's transformers run; its vLLM run
    reads [[0.7483, 0.0756], [0.0888, 0.6300]]). Wrong pooling, a missing or doubled
    end-of-text token, the instruction prefix written differently, or no normalisation
    each moves these by far more than 0.01.
 2. every vector is unit length; a Matryoshka prefix (dimensions 1024) is unit length and
    keeps the same ranking.
 3. two calls name the same fingerprint, and a fingerprint with another scheme is refused
    409 fingerprint_mismatch with nothing embedded.
 4. base64 and base64_float16 decode to the float vectors (to float16's precision).
 5. rerank reproduces the model card: its query against its two documents gives
    logit(score) within 0.5 of 5.0625 and -14.25 (sentence-transformers' CrossEncoder
    logit differences; the score is their sigmoid).
 6. rerank reads the query once: a second document's request reports cached tokens
    (`tokens.cached` > 0 on llama-server; on the Mac the second identical call reports the
    state reused), and the compatible route's top_n=1 is the native route's best.
 7. with --general MODEL: that decide model reranks with Crucible's general template and
    ranks the card's relevant document first.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import sys
import urllib.error
import urllib.request
from typing import Any

EMBED_QUERIES = ["What is the capital of China?", "Explain gravity"]
EMBED_DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to "
    "physical objects and is responsible for the movement of planets around the sun.",
]
EMBED_CARD = [[0.7493, 0.0751], [0.0880, 0.6318]]
RERANK_CARD_LOGITS = [5.0625, -14.25]
TOLERANCE = 0.01
LOGIT_TOLERANCE = 0.5

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)


def post(args: argparse.Namespace, path: str, body: dict[str, Any]) -> tuple[int, Any]:
    request = urllib.request.Request(
        args.url.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.token}",
            "X-Crucible-Api": "1",
            "User-Agent": "check-embed-rerank-live",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def cosine(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in a))


def embed_checks(args: argparse.Namespace) -> None:
    status, queries = post(args, "/v1/embed", {"inputs": EMBED_QUERIES, "input_type": "query"})
    if status != 200:
        check("1 embed answers", False, f"{status} {queries}")
        return
    _, documents = post(args, "/v1/embed", {"inputs": EMBED_DOCUMENTS, "input_type": "document"})
    q, d = queries["embeddings"], documents["embeddings"]
    matrix = [[cosine(a, b) for b in d] for a in q]
    worst = max(abs(matrix[i][j] - EMBED_CARD[i][j]) for i in range(2) for j in range(2))
    check("1 embed reproduces the model card", worst <= TOLERANCE,
          f"{[[round(x, 4) for x in row] for row in matrix]}, worst {worst:.4f}")
    check("2 every vector is unit length", all(abs(norm(v) - 1) < 1e-4 for v in q + d))
    _, short = post(args, "/v1/embed", {"inputs": EMBED_QUERIES, "input_type": "query", "dimensions": 1024})
    _, short_docs = post(args, "/v1/embed", {"inputs": EMBED_DOCUMENTS, "input_type": "document", "dimensions": 1024})
    sm = [[cosine(a, b) for b in short_docs["embeddings"]] for a in short["embeddings"]]
    check("2 a 1024 prefix is unit length and keeps the ranking",
          all(abs(norm(v) - 1) < 1e-4 for v in short["embeddings"])
          and sm[0][0] > sm[0][1] and sm[1][1] > sm[1][0], f"{sm}")
    fingerprint = queries["model"]["fingerprint"]
    check("3 one fingerprint across calls", documents["model"]["fingerprint"] == fingerprint, fingerprint)
    status, refused = post(args, "/v1/embed", {
        "inputs": ["x"], "input_type": "document", "fingerprint": fingerprint[:-1] + "9",
    })
    check("3 another scheme is refused", status == 409
          and refused.get("error", {}).get("code") == "fingerprint_mismatch", f"{status}")
    _, b64 = post(args, "/v1/embed", {"inputs": EMBED_DOCUMENTS[:1], "input_type": "document",
                                       "encoding_format": "base64"})
    _, b16 = post(args, "/v1/embed", {"inputs": EMBED_DOCUMENTS[:1], "input_type": "document",
                                       "encoding_format": "base64_float16"})
    raw32 = base64.b64decode(b64["embeddings"][0])
    raw16 = base64.b64decode(b16["embeddings"][0])
    f32 = struct.unpack(f"<{len(raw32) // 4}f", raw32)
    f16 = struct.unpack(f"<{len(raw16) // 2}e", raw16)
    check("4 the encodings decode to the vector",
          max(abs(a - b) for a, b in zip(f32, d[0])) < 1e-6
          and max(abs(a - b) for a, b in zip(f16, d[0])) < 2e-3)


def logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def rerank_checks(args: argparse.Namespace) -> None:
    body = {"query": EMBED_QUERIES[0], "documents": EMBED_DOCUMENTS}
    status, answer = post(args, "/v1/rerank", body)
    if status != 200:
        check("5 rerank answers", False, f"{status} {answer}")
        return
    logits = [logit(s) for s in answer["scores"]]
    worst = max(abs(a - b) for a, b in zip(logits, RERANK_CARD_LOGITS))
    check("5 rerank reproduces the model card", worst <= LOGIT_TOLERANCE,
          f"logits {[round(x, 3) for x in logits]}, worst {worst:.3f}")
    _, again = post(args, "/v1/rerank", body)
    cached = again["tokens"]["cached"]
    check("6 the query is read from the engine's cache", cached is not None and cached > 0,
          f"tokens {again['tokens']}")
    status, compat = post(args, "/v1/openai/rerank", {**body, "top_n": 1})
    check("6 the compatible route's top_n=1 is the best",
          status == 200 and compat["results"][0]["index"] == answer["results"][0]["index"])


def general_checks(args: argparse.Namespace) -> None:
    status, answer = post(args, "/v1/rerank", {
        "query": EMBED_QUERIES[0], "documents": EMBED_DOCUMENTS, "model": args.general,
    })
    check(f"7 {args.general} reranks with the general template",
          status == 200 and answer["model"]["template"] == "crucible-general-1"
          and answer["scores"][0] > answer["scores"][1], f"{status} {answer.get('scores')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--general", default=None, help="a decide model to rerank with as well")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    embed_checks(args)
    rerank_checks(args)
    if args.general:
        general_checks(args)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
