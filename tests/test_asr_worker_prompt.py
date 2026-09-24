"""`initial_prompt` inside the REAL asr workers, with their libraries stubbed.

`tests/test_asr_api.py` proves the server puts `initial_prompt` in the request
it hands a worker, using the fake workers' recorded requests. That stops at the
envelope. What the key is FOR happens one step further in, in the kwargs each
real worker passes to its library's `transcribe()` — and no fake can prove
that, because a fake never calls `transcribe()`.

So these run `crucible/jobs/asr/worker.py` and `mlx_worker.py` themselves, as
real subprocesses exactly as the server runs them, with three things swapped out
under them and nothing else:

  * `faster_whisper`, or `mlx` + `mlx_whisper`, as tiny stub packages on
    `PYTHONPATH` that record every `transcribe()` call's kwargs to a file and
    count tokens by whitespace — the shapes are the libraries' own
    (faster-whisper 1.2.1 `WhisperModel.hf_tokenizer.encode(...,
    add_special_tokens=False).ids` and `max_length`; mlx-whisper 0.4.3
    `ModelHolder.get_model`, `get_tokenizer(...).encode`, `dims.n_text_ctx`);
  * ffmpeg, as a script that writes N seconds of f32le silence; and
  * a `window_s` of 1, so three seconds of audio is three windows and "every
    window gets the prompt" is something a test can count.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ASR_DIR = Path(__file__).resolve().parents[1] / "crucible" / "jobs" / "asr"
FASTER_WORKER = ASR_DIR / "worker.py"
MLX_WORKER = ASR_DIR / "mlx_worker.py"

SECONDS = 3

#: One stub per library, written into a directory put first on PYTHONPATH.
#: Each records the kwargs of every `transcribe()` call, one JSON line each, to
#: the file named by CRUCIBLE_STUB_CALLS.
FASTER_WHISPER_STUB = '''
import json, os

class _Encoding:
    def __init__(self, ids):
        self.ids = ids

class _HfTokenizer:
    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return _Encoding(list(range(len(text.split()))))

class _Info:
    language = "en"
    language_probability = 0.97

class WhisperModel:
    max_length = 448

    def __init__(self, model_dir, device, compute_type):
        self.hf_tokenizer = _HfTokenizer()

    def transcribe(self, audio, **kwargs):
        with open(os.environ["CRUCIBLE_STUB_CALLS"], "a", encoding="utf-8") as h:
            h.write(json.dumps({"samples": len(audio), **kwargs}) + "\\n")
        return iter(()), _Info()
'''

MLX_CORE_STUB = '''
float16 = "float16"
float32 = "float32"
'''

MLX_WHISPER_INIT_STUB = '''
from .transcribe import transcribe
'''

MLX_WHISPER_TRANSCRIBE_STUB = '''
import json, os

class _Dims:
    n_text_ctx = 448

class _Model:
    is_multilingual = True
    num_languages = 100
    dims = _Dims()

class ModelHolder:
    @classmethod
    def get_model(cls, model_path, dtype):
        return _Model()

def transcribe(audio, **kwargs):
    with open(os.environ["CRUCIBLE_STUB_CALLS"], "a", encoding="utf-8") as h:
        h.write(json.dumps({"samples": len(audio), **kwargs}) + "\\n")
    return {"segments": [], "language": kwargs["language"]}
'''

MLX_WHISPER_TOKENIZER_STUB = '''
class _Tokenizer:
    def encode(self, text):
        return list(range(len(text.split())))

def get_tokenizer(multilingual, *, num_languages=99, language=None, task=None):
    return _Tokenizer()
'''

FFMPEG_STUB = '''
import sys
sys.stdout.buffer.write(b"\\x00" * (16000 * 4 * {seconds}))
sys.stdout.buffer.flush()
'''


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    root = tmp_path / "stubs"
    _write(root / "faster_whisper" / "__init__.py", FASTER_WHISPER_STUB)
    _write(root / "mlx" / "__init__.py", "")
    _write(root / "mlx" / "core.py", MLX_CORE_STUB)
    _write(root / "mlx_whisper" / "__init__.py", MLX_WHISPER_INIT_STUB)
    _write(root / "mlx_whisper" / "transcribe.py", MLX_WHISPER_TRANSCRIBE_STUB)
    _write(root / "mlx_whisper" / "tokenizer.py", MLX_WHISPER_TOKENIZER_STUB)
    return root


@pytest.fixture
def ffmpeg(tmp_path: Path) -> str:
    """An executable the worker can Popen by path, that emits silence."""
    script = tmp_path / "bin" / "ffmpeg_stub.py"
    _write(script, FFMPEG_STUB.format(seconds=SECONDS))
    if os.name == "nt":
        launcher = tmp_path / "bin" / "ffmpeg.cmd"
        _write(launcher, f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        launcher = tmp_path / "bin" / "ffmpeg"
        _write(launcher, f"#!{sys.executable}\n" + script.read_text(encoding="utf-8"))
        launcher.chmod(0o755)
    return str(launcher)


def _request(ffmpeg: str, tmp_path: Path, **overrides: object) -> dict:
    request = {
        "model_dir": str(tmp_path / "weights"),
        "ffmpeg": ffmpeg,
        "audio": str(tmp_path / "audio.m4b"),
        "language": "en",
        "vad_filter": False,
        "word_timestamps": False,
        "initial_prompt": None,
        "device": "cuda",
        "compute_type": "float16",
        "window_s": 1,
        "overlap_s": 0,
    }
    request.update(overrides)
    return request


def _run(
    worker: Path, request: dict, stubs: Path, tmp_path: Path
) -> tuple[list[dict], list[dict]]:
    """The worker's messages, and every `transcribe()` call's kwargs."""
    calls = tmp_path / "calls.jsonl"
    environment = {
        **os.environ,
        "PYTHONPATH": str(stubs),
        "CRUCIBLE_STUB_CALLS": str(calls),
    }
    completed = subprocess.run(
        [sys.executable, str(worker)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    messages = [json.loads(line) for line in completed.stdout.splitlines() if line]
    recorded = (
        [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
        if calls.exists()
        else []
    )
    return messages, recorded


ENGINES = [
    pytest.param(FASTER_WORKER, {"device": "cuda"}, id="faster-whisper"),
    pytest.param(MLX_WORKER, {"device": "metal"}, id="mlx-whisper"),
]


@pytest.mark.parametrize("worker, engine", ENGINES)
def test_the_prompt_reaches_transcribe_in_every_window(
    worker: Path, engine: dict, stubs: Path, ffmpeg: str, tmp_path: Path
) -> None:
    """Every window, not the first: each window is its own `transcribe()` call,
    and each call starts its token history empty."""
    prompt = "Oathbringer. Kaladin, Shallan, Dalinar."
    messages, calls = _run(
        worker, _request(ffmpeg, tmp_path, initial_prompt=prompt, **engine),
        stubs, tmp_path,
    )
    assert messages[-1] == {"type": "done"}, messages
    assert len(calls) == SECONDS
    assert [call["initial_prompt"] for call in calls] == [prompt] * SECONDS


@pytest.mark.parametrize("worker, engine", ENGINES)
def test_no_prompt_reaches_transcribe_as_none(
    worker: Path, engine: dict, stubs: Path, ffmpeg: str, tmp_path: Path
) -> None:
    """None is each library's own default, so passing it is the old call."""
    messages, calls = _run(
        worker, _request(ffmpeg, tmp_path, **engine), stubs, tmp_path
    )
    assert messages[-1] == {"type": "done"}, messages
    assert len(calls) == SECONDS
    assert all(call["initial_prompt"] is None for call in calls)


@pytest.mark.parametrize("worker, engine", ENGINES)
def test_a_missing_key_is_refused_by_name(
    worker: Path, engine: dict, stubs: Path, ffmpeg: str, tmp_path: Path
) -> None:
    request = _request(ffmpeg, tmp_path, **engine)
    del request["initial_prompt"]
    messages, calls = _run(worker, request, stubs, tmp_path)
    assert messages[-1]["type"] == "failed"
    assert "'initial_prompt'" in messages[-1]["message"]
    assert calls == []


@pytest.mark.parametrize("worker, engine", ENGINES)
@pytest.mark.parametrize("value", [5, ["Kaladin"], "  "])
def test_a_prompt_that_is_not_a_nonblank_string_is_refused(
    worker: Path, engine: dict, value: object, stubs: Path, ffmpeg: str, tmp_path: Path
) -> None:
    messages, calls = _run(
        worker, _request(ffmpeg, tmp_path, initial_prompt=value, **engine),
        stubs, tmp_path,
    )
    assert messages[-1]["type"] == "failed"
    assert "initial_prompt" in messages[-1]["message"]
    assert calls == []


@pytest.mark.parametrize("worker, engine", ENGINES)
def test_a_prompt_whisper_would_truncate_is_refused_before_any_window(
    worker: Path, engine: dict, stubs: Path, ffmpeg: str, tmp_path: Path
) -> None:
    """Both libraries keep the LAST `448 // 2 - 1` = 223 prompt tokens; a
    longer prompt would lose its beginning with no error. 223 fits, 224 does
    not (the stubs count one token per word)."""
    fits = " ".join(["name"] * 223)
    messages, calls = _run(
        worker, _request(ffmpeg, tmp_path, initial_prompt=fits, **engine),
        stubs, tmp_path,
    )
    assert messages[-1] == {"type": "done"}, messages
    assert len(calls) == SECONDS

    (tmp_path / "calls.jsonl").unlink()
    too_long = " ".join(["name"] * 224)
    messages, calls = _run(
        worker, _request(ffmpeg, tmp_path, initial_prompt=too_long, **engine),
        stubs, tmp_path,
    )
    assert messages[-1]["type"] == "failed", messages
    assert "224 tokens" in messages[-1]["message"]
    assert "only the last 223" in messages[-1]["message"]
    assert calls == []
