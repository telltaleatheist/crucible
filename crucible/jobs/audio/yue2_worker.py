from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import gc

audiocore = workerio.load_sibling("audiocore", __file__)

LABEL = "yue2"

GIB = 1024**3

YUE2_RESERVE_GIB = 2

SCORE_TOKENS = 4096

SONG_TOKENS = 9000

TOKENS_PER_REPORT = 64

SAMPLE_RATE = 48000

SPANS = (
    ("scoring", 0.2),
    ("composing", 0.55),
    ("synthesizing", 0.15),
    ("decoding", 0.05),
    ("saving", 0.05),
)


def _version_of(distribution: str):
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


class YuE2Engine:
    name = "yue2"
    spans = SPANS

    def __init__(self, request: dict) -> None:
        import torch
        from yue2 import YuE2Pipeline

        self._torch = torch
        model_dir = workerio.require(request, "model_dir", str, LABEL, audiocore.WHY_REQUIRED)
        parts = workerio.require(request, "parts", dict, LABEL, audiocore.WHY_REQUIRED)
        if "vae" not in parts:
            raise RuntimeError(f"the load request names parts {sorted(parts)}; YuE2 needs its 'vae'")
        self.device = workerio.require(request, "device", str, LABEL, audiocore.WHY_REQUIRED)
        budget = workerio.require(request, "memory_budget_bytes", int, LABEL, audiocore.WHY_REQUIRED)
        self._pipe = YuE2Pipeline(
            model_dir,
            parts["vae"],
            device=self.device,
            memory_budget_gib=budget / GIB + YUE2_RESERVE_GIB,
            progress=False,
        )
        self.versions = {
            "yue2-infer": _version_of("yue2-infer"),
            "torch": torch.__version__,
            "transformers": _version_of("transformers"),
        }

    def peak_bytes(self) -> int:
        return int(self._torch.cuda.max_memory_reserved())

    def _close_stage(self, peaks: dict, name: str) -> None:
        peaks[name] = self.peak_bytes()
        gc.collect()
        self._torch.cuda.reset_peak_memory_stats()

    def _stages(self, job, progress, peaks):
        pipe = self._pipe
        stop = lambda: progress.asked_to_stop
        progress.enter("scoring", SCORE_TOKENS)
        ticks = audiocore.Throttled(progress, TOKENS_PER_REPORT)
        plan = pipe.plan(
            job.tags,
            job.lyrics,
            seed=job.seed,
            cfg_scale=job.cfg,
            cancelled=stop,
            on_token=lambda *_: ticks.tick(),
        )
        self._close_stage(peaks, "scoring")
        progress.enter("composing", SONG_TOKENS)
        ticks = audiocore.Throttled(progress, TOKENS_PER_REPORT)
        semantic = pipe.generate_semantic(plan, cancelled=stop, on_token=lambda *_: ticks.tick())
        self._close_stage(peaks, "composing")
        progress.enter("synthesizing")
        latents = pipe.synthesize(semantic, cancelled=stop)
        self._close_stage(peaks, "synthesizing")
        progress.enter("decoding")
        audio = pipe.decode(latents)
        self._close_stage(peaks, "decoding")
        return plan.abc, audio

    def generate(self, job, progress):
        self._torch.cuda.reset_peak_memory_stats()
        peaks: dict = {}
        try:
            score, samples = self._stages(job, progress, peaks)
        except InterruptedError:
            raise audiocore.Cancelled(progress.stage, progress.step) from None
        return audiocore.ArrayAudio(samples, SAMPLE_RATE), score, peaks


ENGINES = {YuE2Engine.name: YuE2Engine}


def main() -> int:
    return audiocore.Worker(LABEL, ENGINES).serve()


if __name__ == "__main__":
    sys.exit(main())
