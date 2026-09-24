"""Drive `crucible.decide`'s reading against a REAL OpenAI-compatible engine, no server.

    python tools/decide_probe.py http://127.0.0.1:8500 qwen3.5-0.8b

The same functions the door calls, in the same order (prime, then the questions), against
the engine's /v1/chat/completions. It exists to answer the one question the fake engine
cannot: that a real engine's reply parses and reads as the contract says. Prints the
answers, label_mass and per-call wall/prompt/cached tokens.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crucible import decide as d  # noqa: E402

base, served = sys.argv[1].rstrip("/"), sys.argv[2]
max_logprobs = int(sys.argv[3]) if len(sys.argv) > 3 else 32

req = d.DecideRequest.model_validate({
    "model": served,
    "state": ("Hi, I was charged twice for my March subscription and the second charge overdrew "
              "my account. I need this fixed today or I'm cancelling."),
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "options": {"billing": "Payment and invoice issues", "technical": "Bugs and errors",
                             "shipping": "Deliveries and returns", "other": "Anything else"}},
        "anger": {"type": "score", "instructions": "How frustrated is the customer?",
                  "levels": ["Calm", "Frustrated but civil", "Very angry"]},
        "urgent": {"type": "yesno", "instructions": "The message conveys urgency"},
        "churn": {"type": "yesno", "instructions": "The customer threatens to leave"},
    },
})
plans = d.plan_all(req)
state_text = d.render_state(req.state)
client = httpx.Client(timeout=120.0)


def post(msgs, k):
    body = d.request_body(served, msgs, k)
    t0 = time.perf_counter()
    r = client.post(f"{base}/v1/chat/completions", json=body)
    wall = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json(), wall


data, wall = post(d.prime_messages(state_text, []), None)
prime = d.read_reply(data, "vllm", want_probs=False)
print(f"prime   wall={wall:6.1f} ms prompt_tokens={prime.prompt_tokens} cached={prime.cached_tokens}")
for item in plans:
    k = d.top_k(len(item.labels), max_logprobs)
    data, wall = post(d.question_messages(state_text, [], item), k)
    reading = d.read_reply(data, "vllm", want_probs=True)
    dist = d.label_distribution(reading.top, item, "vllm")
    ans = d.answer(item, dist, "refuse")
    print(f"{item.name:8s} wall={wall:6.1f} ms prompt_tokens={reading.prompt_tokens} cached={reading.cached_tokens} "
          f"k={k} -> {json.dumps(ans.model_dump(mode='json'))}")
