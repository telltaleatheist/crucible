"""The real Mac video worker (crucible/jobs/video/ltx2mlx_worker.py) against stand-in libraries.

No MLX, no weights, no GPU: `mlx.core` is numpy with a peak counter, and ltx-2-mlx's
`DistilledPipeline` is a stub that records what it was asked and calls the per-step hook the
way the real loops do. What is under test is the worker's own orchestration: the stage order,
one MLX peak per stage, the prompt cache, the start picture's one recompression, the decode
budget, the frames and PCM it hands videocore, and that every component is let go after a job,
a cancelled one included. It runs in a subprocess because importing a worker claims stdout.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1] / "crucible" / "jobs" / "video" / "ltx2mlx_worker.py"

MLX_CORE = textwrap.dedent(
    """
    import numpy

    __version__ = "stub"
    uint8, int16, float32 = numpy.uint8, numpy.int16, numpy.float32
    state = {"cache_limit": None, "peak": 0}

    def set_cache_limit(limit):
        previous, state["cache_limit"] = state["cache_limit"], limit
        return previous

    def clear_cache():
        pass

    def reset_peak_memory():
        state["peak"] = 0

    def get_peak_memory():
        return state["peak"]

    def get_active_memory():
        return 0

    def get_cache_memory():
        return 0

    def eval(*arrays):
        pass

    def use(nbytes):
        state["peak"] = max(state["peak"], nbytes)

    round = numpy.round
    clip = numpy.clip
    contiguous = numpy.ascontiguousarray
    """
)

DISTILLED = textwrap.dedent(
    """
    from types import SimpleNamespace

    import numpy

    import mlx.core as mx

    calls = []


    class Block:
        def __init__(self, name):
            self.name = name

        def free(self):
            calls.append(["free", self.name])


    class PromptEncoder(Block):
        def encode(self, prompt):
            calls.append(["encode", prompt])
            mx.use(20)
            return numpy.zeros((1, 4, 8), numpy.float32), numpy.zeros((1, 4, 4), numpy.float32)


    class Decoder:
        def tiled_decode(self, latent, tiling):
            calls.append(["decode", list(latent.shape), tiling])
            mx.use(15)
            frames = 8 * latent.shape[2] - 7
            height, width = 32 * latent.shape[3], 32 * latent.shape[4]
            yield numpy.full((1, 3, 5, height, width), -1.0, numpy.float32)
            yield numpy.full((1, 3, frames - 5, height, width), 1.0, numpy.float32)


    class VideoDecoderBlock(Block):
        def load(self):
            calls.append(["load", self.name])
            return Decoder()


    class AudioDecoderBlock(Block):
        def __call__(self, latent):
            calls.append(["audio", list(latent.shape)])
            mx.use(2)
            return numpy.full((1, 2, latent.shape[1]), 0.5, numpy.float32)


    class Patchifier:
        def unpatchify(self, tokens, dims):
            return numpy.zeros((1, 128, *dims), numpy.float32)


    class DistilledPipeline:
        def __init__(self, model_dir, gemma_model_id="unused", low_memory=True,
                     low_ram_streaming=False, tile_count=None):
            calls.append(["init", low_memory, low_ram_streaming])
            self._is_25 = True
            self.dit = None
            self.upsampler = None
            self._loaded = False
            self.stepwise = None
            self.prompt_encoder = PromptEncoder("prompt_encoder")
            self.image_conditioner = Block("image_conditioner")
            self.video_decoder_block = VideoDecoderBlock("video_decoder")
            self.audio_decoder_block = AudioDecoderBlock("audio_decoder")
            self.video_patchifier = Patchifier()

        def _load_text_encoder(self):
            calls.append(["pipeline loaded gemma"])

        def _encode_text(self, prompt):
            raise AssertionError("the stub's own encoder ran")

        def _steps(self, stage, steps, dims):
            on_step = self.stepwise.bind(
                latent_frames=dims[0], latent_height=dims[1], latent_width=dims[2],
                decoder_block=self.video_decoder_block, patchifier=self.video_patchifier,
                stage=stage,
            )
            for index in range(steps):
                on_step(index, steps, numpy.zeros(1), 0.5)

        def _stage1(self, prompt, height, width, num_frames, *, frame_rate, seed, stage1_steps,
                    image, images, prompt_relay, generated_keyframes, enable_teacache):
            self._load_text_encoder()
            video, audio = self._encode_text(prompt)
            self.dit, self.upsampler, self._loaded = "dit", "upsampler", True
            calls.append(["stage1", stage1_steps, num_frames, height, width, seed, frame_rate,
                          [[i.path, i.frame_idx, i.strength, i.crf] for i in images or []],
                          list(video.shape), image, prompt_relay, enable_teacache])
            mx.use(24)
            dims = ((num_frames - 1) // 8 + 1, height // 64, width // 64)
            self._steps(1, stage1_steps, dims)
            return SimpleNamespace(video_tokens=None, latent_dims=dims), num_frames, height, width

        def _upsample_latent(self, half):
            calls.append(["upsample", list(half.shape)])
            return half

        def _stage2(self, stage1, upscaled, *, num_frames, frame_rate, seed, stage2_steps,
                    extra_conditionings):
            calls.append(["stage2", stage2_steps, num_frames, seed])
            mx.use(27)
            self.image_conditioner.free()
            self.upsampler = None
            frames, height, width = stage1.latent_dims
            self._steps(2, stage2_steps, (frames, 2 * height, 2 * width))
            samples = round(num_frames / frame_rate * 48000)
            return (numpy.zeros((1, 128, frames, 2 * height, 2 * width), numpy.float32),
                    numpy.zeros((1, samples), numpy.float32))
    """
)

ARGS = textwrap.dedent(
    """
    from typing import NamedTuple


    class ImageConditioningInput(NamedTuple):
        path: str
        frame_idx: int
        strength: float
        crf: int = 33
    """
)

VIDEO_VAE = textwrap.dedent(
    """
    import contextlib

    from ltx_pipelines_mlx.distilled import calls


    def _compute_decode_tiling(latent_shape, frame_rate=24.0, budget_bytes=None):
        calls.append(["tiling", list(latent_shape), frame_rate, budget_bytes])
        return "tiles"


    @contextlib.contextmanager
    def decode_cache_limit():
        calls.append(["cache", "off"])
        yield
        calls.append(["cache", "back"])
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

    model_dir, out_dir, picture = sys.argv[2], sys.argv[3], sys.argv[4] or None
    videocore = worker.videocore
    engine = worker.LtxMlxEngine({"model_dir": model_dir, "device": "metal",
                                  "mlx_cache_limit_bytes": 4000000000})
    report = {"cache_limit": mx.state["cache_limit"], "versions": engine.versions, "runs": []}


    def job(seed, image):
        return videocore.Job({
            "request_id": f"r{seed}", "mode": "text-to-video", "prompt": "a fox in snow",
            "width": 256, "height": 128, "num_frames": 17, "fps": 24, "seed": seed,
            "steps": 8, "refine_steps": 3, "audio": True, "image_path": image,
            "output_path": os.path.join(out_dir, "video.mp4"), "ffmpeg": "ffmpeg",
            "revision": "rev", "backend": "mlx-darwin",
        }, "video")


    def freed():
        pipe = engine._pipe
        return [pipe.dit, pipe.upsampler, pipe._loaded, pipe.stepwise, pipe.embeds]


    for seed, image in ((1, None), (2, picture)):
        before = len(calls)
        clip, peaks, extra = engine.generate(job(seed, image), videocore.Progress(f"r{seed}", engine.spans))
        frames = list(clip.frames)
        report["runs"].append({
            "calls": calls[before:], "peaks": peaks, "extra": extra, "freed": freed(),
            "count": clip.count, "sizes": sorted({len(f) for f in frames}),
            "first": sorted(set(frames[0])), "last": sorted(set(frames[-1])),
            "dark": sum(1 for f in frames if set(f) == {0}),
            "pcm": [len(clip.pcm), clip.channels, clip.sample_rate, clip.pcm[:2].hex()],
            "left": sorted(os.listdir(out_dir)),
        })


    class Stopping(videocore.Progress):
        def reached(self, step):
            if self.stage == "denoising" and step == 3:
                videocore.CANCEL.ask(self.request_id)
            super().reached(step)


    try:
        engine.generate(job(3, None), Stopping("r3", engine.spans))
        report["cancel"] = "not raised"
    except videocore.Cancelled as stopped:
        report["cancel"] = [stopped.stage, stopped.step, freed()]
    print(json.dumps(report), file=sys.stderr)
    """
)


def _stubs(root: Path) -> None:
    files = {
        "mlx/__init__.py": "",
        "mlx/core.py": MLX_CORE,
        "ltx_pipelines_mlx/__init__.py": "",
        "ltx_pipelines_mlx/distilled.py": DISTILLED,
        "ltx_pipelines_mlx/utils/__init__.py": "",
        "ltx_pipelines_mlx/utils/args.py": ARGS,
        "ltx_core_mlx/__init__.py": "",
        "ltx_core_mlx/model/__init__.py": "",
        "ltx_core_mlx/model/video_vae/__init__.py": "",
        "ltx_core_mlx/model/video_vae/video_vae.py": VIDEO_VAE,
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


def _libx264() -> bool:
    try:
        import av

        av.codec.Codec("libx264", "w")
    except Exception:
        return False
    return True


def _run(tmp_path: Path, picture: Path | None) -> tuple[dict, list[dict]]:
    stubs = tmp_path / "stubs"
    _stubs(stubs)
    model_dir = tmp_path / "pack"
    model_dir.mkdir()
    for name in ("embedded_config.json", "quantize_config.json", "text_encoder_config.json",
                 "text_encoder.safetensors", "connector.safetensors", "tokenizer.json",
                 "transformer-distilled.safetensors", "spatial_upscaler_x2_v1_0.safetensors",
                 "vae_encoder_conv.safetensors", "vae_decoder_conv.safetensors",
                 "audio_vae.safetensors", "vocoder.safetensors"):
        (model_dir / name).write_bytes(b"w")
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    done = subprocess.run(
        [sys.executable, "-c", GLUE, str(WORKER), str(model_dir), str(out_dir),
         "" if picture is None else str(picture)],
        env={**__import__("os").environ, "PYTHONPATH": str(stubs)},
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    report = json.loads(done.stderr.strip().splitlines()[-1])
    said = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
    return report, said


@pytest.fixture(scope="module")
def ran(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, list[dict], bool]:
    tmp_path = tmp_path_factory.mktemp("mac-video")
    picture = None
    with_picture = _libx264()
    if with_picture:
        from PIL import Image

        picture = tmp_path / "start.png"
        Image.new("RGB", (300, 200), (30, 160, 90)).save(picture)
    report, said = _run(tmp_path, picture)
    return report, said, with_picture


def test_the_worker_loads_the_pipeline_with_the_manifest_s_cache_limit(ran) -> None:
    report, _, _ = ran
    assert report["cache_limit"] == 4_000_000_000
    assert report["versions"]["mlx"] == "stub"
    first = report["runs"][0]["calls"]
    assert ["pipeline loaded gemma"] not in first


def test_a_clip_runs_the_stages_in_order_with_one_peak_each(ran) -> None:
    report, said, _ = ran
    run = report["runs"][0]
    assert run["peaks"] == {"encoding": 20, "denoising": 24, "refining": 27, "decoding": 15, "audio_decoding": 2}
    stages = []
    for message in said:
        if message.get("type") == "progress" and message["stage"] not in stages[-1:]:
            stages.append(message["stage"])
    assert stages[:5] == ["encoding", "denoising", "refining", "decoding", "audio_decoding"]
    denoise = [m["step"] for m in said if m.get("stage") == "denoising"][:9]
    refine = [m["step"] for m in said if m.get("stage") == "refining"][:4]
    assert denoise == [0, 1, 2, 3, 4, 5, 6, 7, 8] and refine == [0, 1, 2, 3]
    fractions = [m["fraction"] for m in said if m.get("type") == "progress"][: len(denoise) + len(refine) + 3]
    assert fractions == sorted(fractions)


def test_the_pipeline_is_asked_for_8_half_size_steps_then_3_full_size_ones(ran) -> None:
    report, _, _ = ran
    calls = report["runs"][0]["calls"]
    stage1 = next(call for call in calls if call[0] == "stage1")
    assert stage1[1:7] == [8, 17, 128, 256, 1, 24.0]
    assert stage1[7] == [] and stage1[8] == [1, 4, 8]
    assert stage1[9:] == [None, None, False]
    assert ["stage2", 3, 17, 1] in calls
    assert ["upsample", [1, 128, 3, 2, 4]] in calls
    tiling = next(call for call in calls if call[0] == "tiling")
    assert tiling[1] == [1, 128, 3, 4, 8] and tiling[3] == 12 * 1024**3
    order = [call[0] for call in calls if call[0] in ("encode", "stage1", "upsample", "stage2", "decode", "audio")]
    assert order == ["encode", "stage1", "upsample", "stage2", "decode", "audio"]


def test_the_frames_and_the_sound_come_back_for_the_mux(ran) -> None:
    report, _, _ = ran
    run = report["runs"][0]
    assert run["count"] == 17 and run["sizes"] == [256 * 128 * 3]
    assert run["first"] == [0] and run["last"] == [255] and run["dark"] == 5
    samples = round(17 / 24 * 48000)
    assert run["pcm"] == [samples * 2 * 2, 2, 48000, (16384).to_bytes(2, "little").hex()]
    sampling = run["extra"]["sampling"]["passes"]
    assert [(p["size"], p["steps"]) for p in sampling] == [("half", 8), ("full", 3)]
    assert "int8" in run["extra"]["quantization"]["transformer"]


def test_every_component_is_let_go_after_a_job(ran) -> None:
    report, _, _ = ran
    for run in report["runs"]:
        assert run["freed"] == [None, None, False, None, None]
        freed = {call[1] for call in run["calls"] if call[0] == "free"}
        assert {"prompt_encoder", "image_conditioner", "video_decoder", "audio_decoder"} <= freed


def test_the_second_clip_of_a_prompt_skips_the_text_stage(ran) -> None:
    report, _, _ = ran
    first, second = report["runs"]
    assert (first["extra"]["prompt_cache"], second["extra"]["prompt_cache"]) == ("miss", "hit")
    assert "encoding" not in second["peaks"]
    assert not [call for call in second["calls"] if call[0] == "encode"]


def test_a_start_picture_is_recompressed_once_and_handed_over_with_crf_0(ran) -> None:
    report, _, with_picture = ran
    if not with_picture:
        pytest.skip("this PyAV has no libx264; the Mac's pinned av 18.1.0 wheel carries it")
    run = report["runs"][1]
    stage1 = next(call for call in run["calls"] if call[0] == "stage1")
    ((path, frame, strength, crf),) = stage1[7]
    assert path.endswith("video.mp4.start.png") and (frame, strength, crf) == (0, 1.0, 0)
    assert run["left"] == []


def test_a_cancel_between_steps_stops_the_job_and_frees_the_pipeline(ran) -> None:
    report, _, _ = ran
    stage, step, freed = report["cancel"]
    assert (stage, step) == ("denoising", 3)
    assert freed == [None, None, False, None, None]


def _constant(name: str):
    import ast

    for node in ast.parse(WORKER.read_text(encoding="utf-8")).body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) else None
        )
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{WORKER.name} has no {name}")


def test_what_the_worker_needs_is_what_the_manifest_pulls_and_declares() -> None:
    from crucible.videomodels import load_video_manifest

    spec = load_video_manifest("ltx-2.5-distilled").spec("mlx-darwin")
    assert set(_constant("REQUIRED_FILES")) <= set(spec.files)
    assert _constant("AUDIO_SAMPLE_RATE") == spec.audio_sample_rate
    assert _constant("DEFAULT_REFINE_STEPS") == spec.refine_steps
    assert _constant("IMAGE_CRF") == 18
    assert "DECODE_BUDGET_BYTES = 12 * 1024**3" in WORKER.read_text(encoding="utf-8")
    assert 12 * 1024**3 < dict(spec.stage_memory_bytes)["decoding"]
