"""The grammar mlx-lm's patched server enforces, run for real on mlx's CPU backend.

Needs llguidance, mlx, tokenizers and a STOCK mlx-lm 0.31.3, none of which Crucible's
own env carries, so it skips there. To run it, make a throwaway venv (never install
into Crucible's env):

    python3.11 -m venv .venv-grammar
    .venv-grammar/bin/pip install llguidance==1.8.0 'mlx[cpu]==0.32.2' numpy==2.4.6 \\
        tokenizers==0.23.2 transformers==5.17.0 mlx-lm==0.31.3 pytest
    .venv-grammar/bin/python -m pytest -q -p no:cacheprovider --noconftest \\
        tests/test_structured_mlx.py

It imports nothing from Crucible: engines/structured_mlx.py is loaded by path, as the
env carries it, and mlx-lm is a copy patched by Crucible's own scripts. The model is a
2-layer llama with random weights and a toy byte-level BPE tokenizer, so what it writes
is noise; the grammar is what makes it JSON.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

llguidance = pytest.importorskip("llguidance")
mx = pytest.importorskip("mlx.core")
tokenizers = pytest.importorskip("tokenizers")
mlx_lm_spec = importlib.util.find_spec("mlx_lm")
if mlx_lm_spec is None or mlx_lm_spec.origin is None:
    pytest.skip("mlx-lm is not installed", allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
HELPER = ROOT / "crucible" / "engines" / "structured_mlx.py"
PATCHES = ROOT / "crucible" / "envs" / "llm" / "patches"
STOCK_SERVER = "cdfcb4ac848636f9927851a0ec7a951584526530cb7832ba58049e4a9144db8b"
STOCK_GENERATE = "270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f"
MLX_LM_DIR = Path(mlx_lm_spec.origin).parent


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if _sha(MLX_LM_DIR / "server.py") != STOCK_SERVER or _sha(MLX_LM_DIR / "generate.py") != STOCK_GENERATE:
    pytest.skip(
        f"{MLX_LM_DIR} is not stock mlx-lm 0.31.3; these tests patch a copy of it",
        allow_module_level=True,
    )

_spec = importlib.util.spec_from_file_location("crucible_grammar_under_test", HELPER)
grammar = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = grammar
_spec.loader.exec_module(grammar)

SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "n": {"type": "integer", "minimum": 0, "maximum": 9},
        "word": {"type": "string", "maxLength": 6},
    },
    "required": ["ok", "n", "word"],
    "additionalProperties": False,
}
TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: <think>\n"
    "{% if enable_thinking is defined and not enable_thinking %}</think>\n{% endif %}"
    "{% endif %}"
)


def toy_tokenizer() -> Any:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=500,
        special_tokens=["<eos>", "<think>", "</think>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus = ['{"ok": true, "n": 3, "word": "abc"}', '{"tags": ["x", "y"]}', "user assistant"]
    tok.train_from_iterator(corpus * 50, trainer)
    return tok


class FakeHf:
    """What llg_tokenizer reads of a transformers fast tokenizer."""

    def __init__(self, tok: Any) -> None:
        self.backend_tokenizer = tok
        self.eos_token_id = tok.token_to_id("<eos>")
        self._size = tok.get_vocab_size()

    def __len__(self) -> int:
        return self._size


class FakeWrapper:
    """What the processor reads of mlx-lm's TokenizerWrapper."""

    def __init__(self, tok: Any) -> None:
        self._tokenizer = FakeHf(tok)
        self.eos_token_ids = {tok.token_to_id("<eos>")}


@pytest.fixture(scope="module")
def tok() -> Any:
    return toy_tokenizer()


def run(processor: Any, tok: Any, width: int, *, steps: int = 300, seed: int = 0) -> tuple[list[int], Any]:
    """Drive a processor as mlx-lm's generate_step does: the prompt's last token first,
    then each sampled token; random logits, sampled at temperature 1."""
    mx.random.seed(seed)
    eos = tok.token_to_id("<eos>")
    tokens = [tok.token_to_id("Ġ")]
    out: list[int] = []
    for _ in range(steps):
        logits = mx.random.normal((1, width)).astype(mx.bfloat16)
        masked = processor(mx.array(tokens, dtype=mx.int32), logits)
        token = int(mx.random.categorical(masked.astype(mx.float32)).item())
        tokens.append(token)
        out.append(token)
        if token == eos:
            break
    return out, masked


def constraint(kind: str, spec: Any) -> Any:
    return grammar.Constraint(grammar=grammar.compile_grammar(kind, spec), source=kind)


@pytest.mark.parametrize("seed", range(8))
def test_a_sampled_answer_is_the_schema_and_ends_at_its_brace(tok: Any, seed: int) -> None:
    eos = tok.token_to_id("<eos>")
    processor = grammar.GrammarProcessor(constraint("json", SCHEMA), FakeWrapper(tok))
    out, _ = run(processor, tok, tok.get_vocab_size() + 13, steps=2000, seed=seed)
    assert out[-1] == eos, "the answer reached its end"
    text = tok.decode(out[:-1])
    doc = json.loads(text)
    assert set(doc) == {"ok", "n", "word"} and isinstance(doc["ok"], bool)
    assert 0 <= doc["n"] <= 9 and len(doc["word"]) <= 6
    assert text.rstrip().endswith("}"), "nothing after the closing brace but whitespace"


@pytest.mark.parametrize(
    ("kind", "spec", "check"),
    [
        ("choice", ["yes", "no"], lambda text: text in ("yes", "no")),
        ("regex", "[a-c]{3}", lambda text: len(text) == 3 and set(text) <= set("abc")),
        ("json_object", None, lambda text: isinstance(json.loads(text), dict)),
    ],
)
def test_every_kind_vllm_reads_is_kept(tok: Any, kind: str, spec: Any, check: Any) -> None:
    processor = grammar.GrammarProcessor(constraint(kind, spec), FakeWrapper(tok))
    out, _ = run(processor, tok, tok.get_vocab_size(), steps=4000, seed=1)
    assert out[-1] == tok.token_to_id("<eos>")
    assert check(tok.decode(out[:-1]))


def test_eos_is_masked_until_the_grammar_accepts_and_is_all_that_is_left_after(tok: Any) -> None:
    eos = tok.token_to_id("<eos>")
    processor = grammar.GrammarProcessor(constraint("choice", ["yes"]), FakeWrapper(tok))
    width = tok.get_vocab_size()
    logits = mx.zeros((1, width))
    first = processor(mx.array([5], dtype=mx.int32), logits)
    assert first[0, eos].item() == float("-inf")
    tokens = [5]
    for token in tok.encode("yes").ids:
        tokens.append(token)
        step = processor(mx.array(tokens, dtype=mx.int32), logits)
    allowed = [i for i in range(width) if step[0, i].item() != float("-inf")]
    assert allowed == [eos]


def test_logits_wider_than_the_vocabulary_never_offer_the_padding(tok: Any) -> None:
    processor = grammar.GrammarProcessor(constraint("json_object", None), FakeWrapper(tok))
    width = tok.get_vocab_size() + 77
    masked = processor(mx.array([5], dtype=mx.int32), mx.zeros((1, width), dtype=mx.bfloat16))
    assert masked.dtype == mx.bfloat16
    assert all(v == float("-inf") for v in masked[0, tok.get_vocab_size():].tolist())
    assert masked[0, tok.token_to_id("{")].item() == 0


def test_a_failed_matcher_stops_the_answer_and_says_why(tok: Any) -> None:
    eos = tok.token_to_id("<eos>")
    held = constraint("regex", "[a-c]{3}")
    processor = grammar.GrammarProcessor(held, FakeWrapper(tok))
    width = tok.get_vocab_size()
    processor(mx.array([5], dtype=mx.int32), mx.zeros((1, width)))
    forbidden = tok.token_to_id("z")
    masked = processor(mx.array([5, forbidden], dtype=mx.int32), mx.zeros((1, width)))
    assert held.error is not None and "\n" not in held.error
    assert [i for i in range(width) if masked[0, i].item() != float("-inf")] == [eos]


def test_each_sequence_has_its_own_matcher(tok: Any) -> None:
    """Two constrained sequences interleaved step by step, as a batch runs them."""
    eos = tok.token_to_id("<eos>")
    width = tok.get_vocab_size()
    a = grammar.GrammarProcessor(constraint("choice", ["alpha"]), FakeWrapper(tok))
    b = grammar.GrammarProcessor(constraint("regex", "[0-9]{4}"), FakeWrapper(tok))
    mx.random.seed(3)
    streams: dict[str, list[int]] = {"a": [5], "b": [5]}
    done: dict[str, bool] = {"a": False, "b": False}
    for _ in range(200):
        for name, processor in (("a", a), ("b", b)):
            if done[name]:
                continue
            masked = processor(mx.array(streams[name], dtype=mx.int32), mx.random.normal((1, width)))
            token = int(mx.random.categorical(masked).item())
            streams[name].append(token)
            done[name] = token == eos
        if all(done.values()):
            break
    assert tok.decode(streams["a"][1:-1]) == "alpha"
    digits = tok.decode(streams["b"][1:-1])
    assert len(digits) == 4 and digits.isdigit()


def test_the_tokenizer_is_built_once_per_width(tok: Any) -> None:
    wrapper = FakeWrapper(tok)
    one = grammar.llg_tokenizer(wrapper, tok.get_vocab_size())
    assert grammar.llg_tokenizer(wrapper, tok.get_vocab_size()) is one
    assert one.eos_tokens == [tok.token_to_id("<eos>")]
    assert grammar.llg_tokenizer(wrapper, tok.get_vocab_size() + 32).vocab_size == tok.get_vocab_size() + 32


def test_a_schema_that_will_not_compile_is_refused_before_generation() -> None:
    with pytest.raises(grammar.GrammarRefusal) as caught:
        grammar.constraint_of_body({"structured_outputs": {"regex": "(["}})
    assert caught.value.code == "invalid_grammar"
    with pytest.raises(grammar.GrammarRefusal) as caught:
        grammar.constraint_of_body(
            {"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "nonsense"}}}}
        )
    assert caught.value.code == "invalid_grammar"


class Handler(BaseHTTPRequestHandler):
    """An http.server handler with no socket, holding what APIHandler holds."""

    def __init__(self, stream: bool) -> None:
        self.wfile = BytesIO()
        self.request_version = "HTTP/1.1"
        self.requestline = "POST /v1/chat/completions HTTP/1.1"
        self.command = "POST"
        self.client_address = ("127.0.0.1", 0)
        self.stream = stream

    def log_message(self, *_args: Any) -> None:
        return None

    def _set_completion_headers(self, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")


def test_a_whole_answer_whose_matcher_failed_is_a_500_not_the_text() -> None:
    handler = Handler(stream=False)
    handler._set_completion_headers(200)  # handle_completion sends this before generating
    grammar.answer_failure(handler, grammar.Constraint("g", "response_format json_schema", "fuel ran out"))
    sent = handler.wfile.getvalue().decode()
    status_line = sent.split("\r\n", 1)[0]
    assert status_line.split()[1] == "500" and " 200 " not in sent
    body = json.loads(sent.split("\r\n\r\n", 1)[1])
    assert body["error"]["code"] == "structured_output_failed"
    assert "fuel ran out" in body["error"]["message"]


def test_a_streamed_answer_whose_matcher_failed_ends_with_the_error() -> None:
    handler = Handler(stream=True)
    grammar.answer_failure(handler, grammar.Constraint("g", "structured_outputs regex", "boom"))
    events = handler.wfile.getvalue().decode().split("\n\n")
    assert json.loads(events[0][len("data: "):])["error"]["code"] == "structured_output_failed"
    assert events[1] == "data: [DONE]"


# --- mlx-lm itself, patched by Crucible's scripts ---------------------------------


@pytest.fixture(scope="module")
def patched(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of the installed stock mlx-lm, patched by every llm patch script; the
    site-packages directory to put first on PYTHONPATH."""
    env = tmp_path_factory.mktemp("llm-env")
    packages = env / "lib" / "python3.11" / "site-packages"
    shutil.copytree(MLX_LM_DIR, packages / "mlx_lm", ignore=shutil.ignore_patterns("__pycache__"))
    for script in sorted(PATCHES.glob("patch_mlx_lm_*.py")):
        done = subprocess.run([sys.executable, str(script), str(env)], capture_output=True, text=True)
        assert done.returncode == 0, (script.name, done.stdout, done.stderr)
    return packages


@pytest.fixture(scope="module")
def stock(tmp_path_factory: pytest.TempPathFactory) -> Path:
    packages = tmp_path_factory.mktemp("stock") / "site-packages"
    shutil.copytree(MLX_LM_DIR, packages / "mlx_lm", ignore=shutil.ignore_patterns("__pycache__"))
    return packages


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory, tok: Any) -> Path:
    from mlx.utils import tree_flatten
    from mlx_lm.models import llama

    root = tmp_path_factory.mktemp("tiny-llama")
    tok.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<eos>", "chat_template": TEMPLATE}
        ),
        encoding="utf-8",
    )
    config = dict(
        model_type="llama", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, rms_norm_eps=1e-5, vocab_size=tok.get_vocab_size() + 13,
        tie_word_embeddings=True,
    )
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    mx.random.seed(7)
    model = llama.Model(llama.ModelArgs(**config))
    mx.save_safetensors(str(root / "model.safetensors"), dict(tree_flatten(model.parameters())))
    return root


MIXED_BATCH = r'''
import json, sys
import importlib.util
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import BatchGenerator
spec = importlib.util.spec_from_file_location("g", sys.argv[2]); g = importlib.util.module_from_spec(spec)
sys.modules["g"] = g; spec.loader.exec_module(g)
model, tokenizer = load(sys.argv[1])
prompt = tokenizer.encode("user: hi" + chr(10) + "assistant: ")
eos = set(tokenizer.eos_token_ids)
out = {}

def grammar_row():
    held = g.Constraint(grammar=g.compile_grammar("regex", "[a-c]{5}"), source="regex")
    return [[g.GrammarProcessor(held, tokenizer)]]

def step(gen, times):
    for _ in range(times):
        for r in gen.next()[1]:
            row = out.setdefault(r.uid, [])
            if r.finish_reason is None and not (row and row[-1] in eos):
                row.append(r.token)

def text(uid):
    tokens = out[uid]
    ends = [i for i, t in enumerate(tokens) if t in eos]
    return tokenizer.decode(tokens[: ends[0]] if ends else tokens)

# Two requests a moment apart: the plain one is in the prompt batch when the
# constrained one joins it (PromptProcessingBatch.extend's None).
first = BatchGenerator(model, completion_batch_size=4, prefill_batch_size=4)
first.insert([prompt], max_tokens=[20], logits_processors=[[]])
first.next()
(together,) = first.insert([prompt], max_tokens=[20], logits_processors=grammar_row())
step(first, 40)
# A plain request finishes while another plain one runs on, and a constrained one
# joins after (GenerationBatch.filter's stale list).
second = BatchGenerator(model, completion_batch_size=4, prefill_batch_size=4)
second.insert([prompt, prompt], max_tokens=[2, 60], logits_processors=[[], []])
step(second, 6)
(after,) = second.insert([prompt], max_tokens=[20], logits_processors=grammar_row())
step(second, 40)
print(json.dumps({"together": text(together), "after": text(after)}))
'''


def _python(packages: Path, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(packages)},
    )


def test_a_plain_and_a_constrained_sequence_share_a_batch(patched: Path, weights: Path) -> None:
    done = _python(patched, "-c", MIXED_BATCH, str(weights), str(HELPER))
    assert done.returncode == 0, done.stderr[-3000:]
    texts = json.loads(done.stdout.strip().splitlines()[-1])
    for name, text in texts.items():
        assert len(text) == 5 and set(text) <= set("abc"), (name, text)


def test_stock_mlx_lm_dies_on_that_batch(stock: Path, weights: Path) -> None:
    """Why the batch patch exists: the falsifying run for the one above."""
    done = _python(stock, "-c", MIXED_BATCH, str(weights), str(HELPER))
    assert done.returncode != 0
    assert "TypeError: 'NoneType' object is not iterable" in done.stderr


@pytest.fixture(scope="module")
def server(patched: Path, weights: Path, tmp_path_factory: pytest.TempPathFactory) -> Any:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    log = tmp_path_factory.mktemp("server") / "server.log"
    with open(log, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "mlx_lm", "server", "--model", str(weights), "--port", str(port),
             "--decode-concurrency", "8", "--prompt-concurrency", "8", "--prompt-cache-size", "8"],
            stdout=handle, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(patched)},
        )
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        yield (f"http://127.0.0.1:{port}/v1/chat/completions", str(weights), log)
    finally:
        process.terminate()
        process.wait(30)
    assert "Traceback" not in log.read_text(encoding="utf-8"), log.read_text(encoding="utf-8")[-3000:]


def post(server: Any, body: dict[str, Any]) -> tuple[int, Any]:
    url, model, _ = server
    data = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "go"}], "max_tokens": 800, **body}
    ).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read().decode()
            return response.status, raw if body.get("stream") else json.loads(raw)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode())


def quiet_whitespace(tok: Any) -> dict[str, float]:
    """A random model left to itself writes whitespace for as long as the JSON
    grammar lets it (whitespace_flexible, as vLLM compiles it); bias it away so a
    run ends in a few dozen tokens. The bias is applied before the mask."""
    return {str(tok.token_to_id(ch)): -8.0 for ch in ("Ġ", "Ċ", "ĉ", "č")}


SCHEMA_FORMAT = {"type": "json_schema", "json_schema": {"name": "v", "schema": SCHEMA, "strict": True}}


def test_constrained_and_plain_chats_at_once(server: Any, tok: Any) -> None:
    bias = quiet_whitespace(tok)
    cases = {
        "plain": {"temperature": 1.0, "max_tokens": 40},
        "schema": {"temperature": 1.0, "logit_bias": bias, "response_format": SCHEMA_FORMAT},
        "schema_thinking_off": {
            "temperature": 1.0, "logit_bias": bias, "response_format": SCHEMA_FORMAT,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "choice": {"temperature": 1.0, "structured_outputs": {"choice": ["yes", "no"]}},
        "regex": {"temperature": 1.0, "structured_outputs": {"regex": "[a-c]{3}"}},
        "seeded": {"temperature": 1.0, "seed": 3, "logit_bias": bias, "response_format": SCHEMA_FORMAT},
    }
    results: dict[str, Any] = {}
    threads = [
        threading.Thread(target=lambda n=name, b=body: results.__setitem__(n, post(server, b)))
        for name, body in cases.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for name, (status, body) in results.items():
        assert status == 200, (name, body)
        message = body["choices"][0]["message"]
        if name == "plain":
            continue
        assert body["choices"][0]["finish_reason"] == "stop", (name, message)
        content = message.get("content", "")
        assert not message.get("reasoning"), (name, "a constrained answer is the answer, not reasoning")
        if name in ("schema", "schema_thinking_off", "seeded"):
            doc = json.loads(content)
            assert set(doc) == {"ok", "n", "word"}, (name, doc)
        elif name == "choice":
            assert content in ("yes", "no")
        else:
            assert len(content) == 3 and set(content) <= set("abc")


def test_a_streamed_constrained_answer(server: Any, tok: Any) -> None:
    status, raw = post(
        server,
        {"stream": True, "temperature": 1.0, "logit_bias": quiet_whitespace(tok), "response_format": SCHEMA_FORMAT},
    )
    assert status == 200
    chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices"))
    assert set(json.loads(text)) == {"ok", "n", "word"}


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"guided_json": SCHEMA}, "structured_output_not_served"),
        ({"response_format": {"type": "structural_tag"}}, "structured_output_not_served"),
        ({"response_format": {"type": "json_schema", "json_schema": {"name": "v"}}}, "invalid_response_format"),
        ({"structured_outputs": {"regex": "(["}}, "invalid_grammar"),
    ],
)
def test_the_server_refuses_by_name(server: Any, body: dict[str, Any], code: str) -> None:
    status, answer = post(server, body)
    assert status == 400 and answer["error"]["code"] == code
