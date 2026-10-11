"""The real score worker with SheetSage2 replaced by a stand-in.

Everything but the model runs for real: workerio, the request, numpy reading the decoded
samples, SheetSage2's progress callback (and so the cancel check), the files, and the
YuE2-native pass with the vendored yue2music tools. The stand-in returns a result shaped
like pipeline_sheetsage2.transcribe's, its windows planned the way SheetSage2 plans them,
and a score from tests/score_fixtures.py (or the file CRUCIBLE_FAKE_SCORE_ABC names).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REAL_WORKER = HERE.parent / "crucible" / "jobs" / "score" / "worker.py"

sys.path.insert(0, str(HERE))
import score_fixtures  # noqa: E402

sys.path.pop(0)


def _load_real_worker():
    spec = importlib.util.spec_from_file_location("crucible_score_worker", REAL_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = _load_real_worker()

CAP = 5120

WINDOW_S, OVERLAP_S, LOOKAHEAD_S = 300.0, 200.0, 100.0

# MThd and an empty track: enough to be a MIDI file.
MIDI_BYTES = b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0MTrk\x00\x00\x00\x04\x00\xff\x2f\x00"


def _transcribe(request: dict) -> None:
    transcript = os.environ.get("CRUCIBLE_FAKE_SCORE_TRANSCRIPT")
    if transcript:
        with open(transcript, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(request) + "\n")


def plan(duration: float) -> list[dict]:
    """pipeline_sheetsage2.sliding_window_plan with its default preset."""
    start, accepted, rows = 0.0, 0.0, []
    while True:
        last = start + WINDOW_S >= duration - 1e-6
        accept_end = duration if last else start + WINDOW_S - LOOKAHEAD_S
        rows.append({"start": start, "end": min(duration, start + WINDOW_S),
                     "accept_start": accepted, "accept_end": accept_end})
        if last:
            return rows
        accepted = accept_end
        start = min(start + WINDOW_S - OVERLAP_S, duration - WINDOW_S)


class FakeEngine:
    name = "sheetsage2"

    def __init__(self, request: dict) -> None:
        _transcribe({**request, "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                     "hf_offline": os.environ.get("HF_HUB_OFFLINE")})
        if os.environ.get("CRUCIBLE_FAKE_SCORE_LOAD_FAIL") == "1":
            raise RuntimeError("fake score worker was told not to load")
        self.device = request["device"]
        self.autocast = request["autocast"]
        self.cap = CAP
        self.versions = {"fake": "1.0"}

    @property
    def dtype(self) -> str:
        return "bf16" if self.device == "cuda" and self.autocast == "bfloat16" else "fp32"

    def _abc(self, job) -> str | None:
        if os.environ.get("CRUCIBLE_FAKE_SCORE_NO_ABC") == "1":
            return None
        named = os.environ.get("CRUCIBLE_FAKE_SCORE_ABC")
        if named:
            return Path(named).read_text(encoding="utf-8")
        return score_fixtures.SHEETSAGE2_SONG if job.keep_harmony else score_fixtures.MELODY_ONLY_SONG

    def transcribe(self, waveform, job, progress) -> dict:
        _transcribe({"op": "score", "request_id": job.request_id, "samples": int(waveform.size),
                     "prompts": job.prompts, "melody_only": not job.keep_harmony,
                     "dtype": self.dtype})
        report = progress.sheetsage2(self.cap)
        duration = waveform.size / worker.SAMPLE_RATE
        windows = plan(duration)
        tokens = int(os.environ.get("CRUCIBLE_FAKE_SCORE_TOKENS") or 900)
        pause = float(os.environ.get("CRUCIBLE_FAKE_SCORE_PAUSE_S") or 0)
        warnings, records = [], []
        report({"stage": "audio"})
        for index, window in enumerate(windows):
            report({"stage": "encoding", "window": index + 1, "windows": len(windows),
                    "start": window["start"]})
            deadline = time.monotonic() + pause
            for made in range(64, tokens + 1, 64):
                report({"stage": "decoding", "window": index + 1, "windows": len(windows),
                        "tokens": made})
                if time.monotonic() < deadline:
                    time.sleep(0.02)
            while time.monotonic() < deadline:
                report({"stage": "decoding", "window": index + 1, "windows": len(windows),
                        "tokens": tokens})
                time.sleep(0.02)
            if tokens > self.cap:
                warnings.append(f"Window {index + 1} reached the token limit; inspect its token coverage")
            records.append({**window, "window_index": index, "prefix_tokens": 0 if index == 0 else 40,
                            "tokens": tokens, "events": 300, "accepted_events": 200,
                            "elapsed_seconds": 1.5})
        report({"stage": "notation"})
        abc = self._abc(job)
        os.makedirs(job.output_dir, exist_ok=True)
        Path(job.output_dir, "transcription.mid").write_bytes(MIDI_BYTES)
        if abc is not None:
            Path(job.output_dir, "score.abc").write_text(abc, encoding="utf-8")
        report({"stage": "complete"})
        return {
            "audio": "audio", "duration_seconds": duration, "prompts": job.prompts,
            "preset": "default", "melody_only": not job.keep_harmony, "dtype": self.dtype,
            "window_seconds": WINDOW_S, "overlap_seconds": OVERLAP_S, "lookahead_seconds": LOOKAHEAD_S,
            "windows": records, "elapsed_seconds": 3.25, "warnings": warnings,
            "peak_gpu_mib": 3071.5 if self.device == "cuda" else 0,
            "melody_notes": 33, "vocal_notes": 29, "instrumental_notes": 4, "num_events": 600,
            "abc_measures": 15, "abc_error": None if abc is not None else "No key was decoded; cannot construct a keyed ABC score",
            "diagnostics": ["moved 1 overlapping note"], "abc": abc, "midi": MIDI_BYTES,
        }

    def peak_bytes(self) -> int:
        return 11

    def release(self) -> None:
        return None


worker.ENGINES["sheetsage2"] = FakeEngine


if __name__ == "__main__":
    sys.exit(worker.main())
