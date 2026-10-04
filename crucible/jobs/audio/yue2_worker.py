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


# The Mac's proof that bfloat16 causal attention is causal, run on every load before a
# token is generated. torch <= 2.12.1 on Apple's Metal backend applied `is_causal` per
# block of four query rows, so a query saw up to three FUTURE keys - silently, no error,
# no NaN (YuE issue #176, pytorch#195910); YuE2 is bfloat16-only and prefills through that
# kernel. The env pins a torch that fixed it, but whether a given chip and macOS are sound
# is a measured fact, not a version string: the bug showed on an M4 Pro and M5 Max and not
# on this M1 Ultra. The check is YuE2's own `sdpa` at its attention shape (16 query heads,
# 8 KV heads, head_dim 128) against an explicit lower-triangular mask: correct kernels agree
# to bfloat16 rounding (~0.003 relative), the leaking one is off by 0.3-0.6.
CAUSAL_CHECK_LENGTHS = (17, 128, 705)
CAUSAL_CHECK_LIMIT = 0.02


def mps_causal_is_sound(torch, sdpa, device: str = "mps") -> list[tuple[int, float]]:
    """Relative error of `sdpa(..., is_causal=True)` against an explicit causal mask, per
    length; raises when any is past CAUSAL_CHECK_LIMIT."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    errors = []
    for length in CAUSAL_CHECK_LENGTHS:
        shape_q, shape_kv = (1, 16, length, 128), (1, 8, length, 128)
        q, k, v = (torch.randn(shape, generator=generator).to(device, torch.bfloat16)
                   for shape in (shape_q, shape_kv, shape_kv))
        mask = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        causal = sdpa(q, k, v, is_causal=True).float()
        explicit = sdpa(q, k, v, attn_mask=mask).float()
        error = float((causal - explicit).norm() / explicit.norm())
        errors.append((length, error))
    leaking = [(n, e) for n, e in errors if not e <= CAUSAL_CHECK_LIMIT]
    if leaking:
        raise RuntimeError(
            f"bfloat16 causal attention on {device} is not causal with torch "
            f"{torch.__version__}: is_causal differs from an explicit causal mask by "
            + ", ".join(f"{e:.3f} at length {n}" for n, e in leaking)
            + f" (limit {CAUSAL_CHECK_LIMIT}). A query would see future tokens and YuE2 would "
            "render a song from a corrupted prompt without any error. This is YuE issue "
            "#176 / pytorch#195910, fixed in torch 2.13; the env should pin one that "
            "fixed it - reinstall it with `crucible install audio`, and if it persists "
            "this chip and macOS need a torch newer than the recipe's"
        )
    return errors


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
        self.causal_check = None
        if self.device == "mps":
            from yue2.modeling_yue2 import sdpa

            self.causal_check = mps_causal_is_sound(torch, sdpa)
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
        if self.causal_check is not None:
            self.versions["mps_causal_check"] = {
                str(length): round(error, 4) for length, error in self.causal_check
            }

    def peak_bytes(self) -> int:
        if self.device == "cuda":
            return int(self._torch.cuda.max_memory_reserved())
        # Metal keeps no peak: what the driver holds at a stage's end, before it is
        # released, is the nearest honest figure (the Stable Audio worker reads the same).
        return int(self._torch.mps.driver_allocated_memory())

    def _reset_peak(self) -> None:
        if self.device == "cuda":
            self._torch.cuda.reset_peak_memory_stats()
        else:
            self._torch.mps.empty_cache()

    def _close_stage(self, peaks: dict, name: str) -> None:
        peaks[name] = self.peak_bytes()
        gc.collect()
        self._reset_peak()

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
        self._reset_peak()
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
