"""The Mac worker's lip-dub engine (ltx2mlx_worker.LipDubEngine) against stand-in libraries.

The same stand-ins as test_video_mac_worker.py, plus a `LipDubPipeline` stub that records what
it was built with and asked for, and starts its two passes through `_stepwise_hook` the way
ltx_pipelines_mlx/lipdub.py does. Under test: the load (the IC-LoRA at 1.0, the text encoder's
folder, block streaming always on), the request handed to `generate_lipdub`, one stage and one
MLX peak per pass, the decode and the sound handed to the mux at the reference's exact rate,
the level it reports, and a cancel or a disagreement letting everything go.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from .test_video_mac_worker import WORKER, _stubs

LIPDUB = textwrap.dedent(
    """
    import numpy

    import mlx.core as mx
    from ltx_pipelines_mlx.distilled import DistilledPipeline, calls


    class LipDubPipeline(DistilledPipeline):
        def __init__(self, model_dir, lora_paths=None, gemma_model_id="unused",
                     low_memory=True, low_ram_streaming=False):
            super().__init__(model_dir, low_memory=low_memory, low_ram_streaming=low_ram_streaming)
            calls.append(["lipdub init", [list(p) for p in lora_paths], gemma_model_id])

        def generate_lipdub(self, prompt, reference_video_path, height=480, width=704,
                            reference_strength=1.0, images=None, seed=42, stage1_steps=None,
                            stage2_steps=None):
            calls.append(["generate_lipdub", prompt, reference_video_path, height, width,
                          reference_strength, seed, stage1_steps, stage2_steps])
            frames = int(open(reference_video_path).read())
            mx.use(14)
            latent_frames = (frames - 1) // 8 + 1
            for stage, steps, scale, peak in ((1, stage1_steps, 64, 15), (2, stage2_steps, 32, 16)):
                on_step = self._stepwise_hook(latent_frames, height // scale, width // scale,
                                              stage=stage)
                mx.use(peak)
                for index in range(steps):
                    on_step(index, steps, numpy.zeros(1), 0.5)
            video = numpy.zeros((1, 128, latent_frames, height // 32, width // 32), numpy.float32)
            return video, numpy.zeros((1, 8, 25, 16), numpy.float32), 30000 / 1001
    """
)

GLUE = textwrap.dedent(
    """
    import importlib.util, json, os, sys

    spec = importlib.util.spec_from_file_location("ltx2mlx_worker", sys.argv[1])
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    import mlx.core as mx
    from ltx_pipelines_mlx.distilled import calls

    model_dir, out_dir, reference = sys.argv[2], sys.argv[3], sys.argv[4]
    videocore = worker.videocore
    load = {"engine": "ltx-2-mlx", "pipeline": "lip-dub", "model_dir": model_dir,
            "device": "metal", "mlx_cache_limit_bytes": 4000000000,
            "lora_path": os.path.join(model_dir, "ic-lora", "dubit.safetensors"),
            "text_encoder_dir": os.path.join(model_dir, "text-encoder"), "desktop": None}
    engine = worker.engine_for(load)
    report = {"type": type(engine).__name__, "spans": [n for n, _ in engine.spans],
              "load_calls": list(calls)}
    calls.clear()


    def job(seed, frames=193):
        return videocore.Job({
            "request_id": f"r{seed}", "mode": "lip-dub", "prompt": 'he says: "hello"',
            "width": 768, "height": 448, "num_frames": frames, "fps": 30000 / 1001,
            "frame_rate": "30000/1001", "seed": seed, "steps": 8, "refine_steps": 3,
            "audio": True, "image_path": None, "soundtrack_path": None,
            "reference_path": reference, "reference_strength": 0.8,
            "output_path": os.path.join(out_dir, "video.mp4"), "ffmpeg": "ffmpeg",
            "revision": "rev", "backend": "mlx-darwin",
        }, "video")


    def freed():
        pipe = engine._pipe
        return [pipe.dit, pipe.upsampler, pipe._loaded, pipe.marks]


    class Watching(videocore.Progress):
        def enter(self, stage, steps=None):
            report.setdefault("entered", []).append([stage, steps])
            super().enter(stage, steps)


    clip, peaks, extra = engine.generate(job(1), Watching("r1", engine.spans))
    frames = list(clip.frames)
    report["run"] = {"calls": list(calls), "peaks": peaks, "extra": extra, "count": clip.count,
                     "rate": clip.rate, "fps": clip.fps, "sizes": sorted({len(f) for f in frames}),
                     "pcm": [len(clip.pcm), clip.channels, clip.sample_rate], "freed": freed()}


    class Stopping(videocore.Progress):
        def reached(self, step):
            if self.stage == "denoising" and step == 2:
                videocore.CANCEL.ask(self.request_id)
            super().reached(step)


    try:
        engine.generate(job(2), Stopping("r2", engine.spans))
        report["cancel"] = "not raised"
    except videocore.Cancelled as stopped:
        report["cancel"] = [stopped.stage, stopped.step, freed()]
    try:
        engine.generate(job(3, frames=97), videocore.Progress("r3", engine.spans))
        report["disagree"] = "not raised"
    except RuntimeError as exc:
        report["disagree"] = [str(exc), freed()]
    distilled = worker.engine_for({**load, "pipeline": "distilled"})
    report["distilled"] = type(distilled).__name__
    report["level"] = worker.level_dbfs((b"\\x00\\x40" * 4) + (b"\\x00\\xc0" * 4))
    print(json.dumps(report), file=sys.stderr)
    """
)


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp_path = tmp_path_factory.mktemp("lipdub-worker")
    stubs = tmp_path / "stubs"
    _stubs(stubs)
    (stubs / "ltx_pipelines_mlx" / "lipdub.py").write_text(LIPDUB, encoding="utf-8")
    model_dir = tmp_path / "pack"
    for name in ("embedded_config.json", "quantize_config.json", "connector.safetensors",
                 "transformer-distilled-1.1.safetensors", "spatial_upscaler_x2_v1_1.safetensors",
                 "spatial_upscaler_x2_v1_1_config.json", "vae_encoder.safetensors",
                 "vae_decoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors",
                 "ic-lora/dubit.safetensors", "text-encoder/config.json",
                 # the distilled engine's own pack files, for the dispatch check
                 "text_encoder_config.json", "text_encoder.safetensors", "tokenizer.json",
                 "transformer-distilled.safetensors", "spatial_upscaler_x2_v1_0.safetensors",
                 "vae_encoder_conv.safetensors", "vae_decoder_conv.safetensors"):
        (model_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (model_dir / name).write_bytes(b"w")
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    reference = tmp_path / "talk.mp4"
    reference.write_text("195", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-c", GLUE, str(WORKER), str(model_dir), str(out_dir), str(reference)],
        env={**__import__("os").environ, "PYTHONPATH": str(stubs)},
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stderr.strip().splitlines()[-1])


def test_the_lip_dub_pipeline_is_built_with_the_lora_the_text_encoder_and_streaming(ran) -> None:
    assert ran["type"] == "LipDubEngine" and ran["distilled"] == "LtxMlxEngine"
    init, lipdub = ran["load_calls"]
    assert init == ["init", True, True]
    assert lipdub[1][0][0].endswith("ic-lora/dubit.safetensors") and lipdub[1][0][1] == 1.0
    assert lipdub[2].endswith("text-encoder")
    assert ran["spans"] == ["encoding", "denoising", "refining", "decoding", "audio_decoding", "muxing"]


def test_the_request_reaches_generate_lipdub_and_each_pass_is_its_own_stage(ran) -> None:
    run = ran["run"]
    asked = next(call for call in run["calls"] if call[0] == "generate_lipdub")
    assert asked[3:] == [448, 768, 0.8, 1, 8, 3] and asked[2].endswith("talk.mp4")
    # muxing is the videocore Worker's own stage, after generate returns
    assert ran["entered"] == [
        ["encoding", None], ["denoising", 8], ["refining", 3], ["decoding", None],
        ["audio_decoding", None],
    ]
    assert run["peaks"] == {"encoding": 14, "denoising": 15, "refining": 16, "decoding": 15,
                            "audio_decoding": 2}


def test_the_clip_keeps_the_reference_s_exact_rate_and_reports_its_level(ran) -> None:
    run = ran["run"]
    assert (run["count"], run["rate"], run["sizes"]) == (193, "30000/1001", [768 * 448 * 3])
    assert run["pcm"][1:] == [2, 48000]
    extra = run["extra"]
    assert extra["lora_strength"] == 1.0 and extra["soundtrack"] is None
    assert [p["steps"] for p in extra["sampling"]["passes"]] == [8, 3]
    assert ran["level"] == {"mean": -6.0, "peak": -6.0}
    assert run["freed"] == [None, None, False, None]


def test_a_cancel_or_a_disagreeing_reference_lets_everything_go(ran) -> None:
    stage, step, freed = ran["cancel"]
    assert (stage, step, freed) == ("denoising", 2, [None, None, False, None])
    said, freed = ran["disagree"]
    assert "the two probes of the reference must agree" in said
    assert freed == [None, None, False, None]
