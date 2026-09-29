"""The stable-audio worker puts only the float16 model on the card.

stable_audio_3's own loader moves the float32 model to the device and halves it
there. For Medium that is 10.4 GB against a 7.45 GiB per-process cap, and the
first real job on the PC died in the load (2026-09-29). The worker loads on the
CPU, halves there, then moves. Stubs only: no torch, no model, no GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "crucible" / "jobs" / "audio" / "stable_audio_worker.py"

TORCH_STUB = '''
float16 = "float16"
float32 = "float32"
__version__ = "stub"

class _Props:
    total_memory = 24 * 1024 ** 3

class cuda:
    @staticmethod
    def is_available():
        return True
    @staticmethod
    def get_device_properties(index):
        return _Props()
    @staticmethod
    def set_per_process_memory_fraction(fraction, index):
        pass
'''

LOADING_STUB = '''
import os
from stable_audio_3 import calls

class FakeModel:
    def __init__(self):
        self.device = None
        self.dtype = "float32"
    def to(self, target):
        if target in ("float16", "float32"):
            calls.append(("cast", target, self.device))
            self.dtype = target
        else:
            calls.append(("move", target, self.dtype))
            self.device = target
        return self
    def eval(self):
        return self
    def requires_grad_(self, flag):
        return self

def load_diffusion_cond(model_config, ckpt_path, device="cuda", model_half=False):
    # Mirrors the real package: move to the device, then halve.
    model = FakeModel()
    model.to(device).eval().requires_grad_(False)
    if model_half:
        model.to("float16")
    return model
'''

MODEL_STUB = '''
class StableAudioModel:
    def __init__(self, model, config, device, half):
        self.model = model
'''


def _stubs(root: Path) -> None:
    (root / "torch.py").write_text(TORCH_STUB, encoding="utf-8")
    package = root / "stable_audio_3"
    (package / "models").mkdir(parents=True)
    (package / "__init__.py").write_text("calls = []\n", encoding="utf-8")
    (package / "loading_utils.py").write_text(LOADING_STUB, encoding="utf-8")
    (package / "model.py").write_text(MODEL_STUB, encoding="utf-8")
    (package / "models" / "__init__.py").write_text("", encoding="utf-8")
    (package / "models" / "transformer.py").write_text("flash_attn_func = object()\n", encoding="utf-8")


def test_only_the_float16_model_reaches_the_card(tmp_path: Path) -> None:
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _stubs(stubs)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_config.json").write_text(json.dumps({"sample_rate": 44100}), encoding="utf-8")

    script = f'''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("stable_audio_worker", {str(WORKER)!r})
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker.StableAudio3Engine({{
    "model_dir": {str(model_dir)!r},
    "device": "cuda",
    "dtype": "float16",
    "memory_cap_bytes": 8000000000,
}})
import stable_audio_3
print(json.dumps(stable_audio_3.calls), file=sys.stderr)
'''
    env = {**os.environ, "PYTHONPATH": str(stubs)}
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0, done.stderr
    calls = [tuple(call) for call in json.loads(done.stderr.strip().splitlines()[-1])]
    moves_to_card = [call for call in calls if call[0] == "move" and call[1] == "cuda"]
    assert moves_to_card, calls
    assert all(dtype == "float16" for _, _, dtype in moves_to_card), calls
