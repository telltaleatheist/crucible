from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import gc
import json

from workerio import cap_memory

audiocore = workerio.load_sibling("audiocore", __file__)

LABEL = "stable audio"

T5GEMMA_DIR = "t5gemma-b-b-ul2"

SPANS = (("encoding", 0.05), ("denoising", 0.8), ("decoding", 0.1), ("saving", 0.05))
# How much shorter than asked the audio may come back (one latent frame and rounding)
# before it is not the request.
SHORT_TOLERANCE_S = 0.1


def localise_text_encoder(node, t5_dir: str):
    if isinstance(node, list):
        return [localise_text_encoder(item, t5_dir) for item in node]
    if not isinstance(node, dict):
        return node
    found = {key: localise_text_encoder(value, t5_dir) for key, value in node.items()}
    names_t5gemma = str(found.get("model_name", "")).startswith("google/t5gemma")
    if "repo_id" in found or names_t5gemma:
        found.pop("repo_id", None)
        found.pop("subfolder", None)
        found["model_path"] = t5_dir
    return found


def require_flash_attention() -> None:
    from stable_audio_3.models import transformer

    if transformer.flash_attn_func is None:
        raise RuntimeError(
            "flash-attn does not import in this env, and on CUDA Stable Audio 3 Medium "
            "turns that into static instead of sound (Stability's README). Rebuild the env "
            "with `crucible install audio --force`"
        )


def _version_of(distribution: str):
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


class StableAudio3Engine:
    name = "stable-audio-3"
    notes = None
    # A diffusion model: no stage decodes tokens, so none can run to a cap.
    decode_stages = None
    # Nothing is kept in host memory on purpose (yue2_worker.HostHomes is YuE2's).
    host_homes = None
    spans = SPANS

    def __init__(self, request: dict) -> None:
        import torch
        from stable_audio_3.loading_utils import load_diffusion_cond
        from stable_audio_3.model import StableAudioModel

        self._torch = torch
        model_dir = workerio.require(request, "model_dir", str, LABEL, audiocore.WHY_REQUIRED)
        self.device = workerio.require(request, "device", str, LABEL, audiocore.WHY_REQUIRED)
        half = workerio.require(request, "dtype", str, LABEL, audiocore.WHY_REQUIRED) == "float16"
        if self.device == "cuda":
            cap_memory(torch, request.get("memory_cap_bytes"))
            require_flash_attention()
        with open(os.path.join(model_dir, "model_config.json"), encoding="utf-8") as handle:
            config = localise_text_encoder(json.load(handle), os.path.join(model_dir, T5GEMMA_DIR))
        # The package's loader moves the float32 model to the device and only then
        # halves it, so Medium's 10.4 GB of float32 weights met the card whole and
        # overran its 7.45 GiB cap before any sound was made (PC, 2026-09-29). Load
        # and halve in host memory; only the float16 copy goes to the card.
        model = load_diffusion_cond(
            config, os.path.join(model_dir, "model.safetensors"), device="cpu", model_half=half
        )
        model.to(self.device)
        model.use_lora = False
        model.lora_names = []
        self._model = StableAudioModel(model, config, self.device, half)
        self.sample_rate = int(config["sample_rate"])
        self.versions = {
            "stable-audio-3": _version_of("stable-audio-3"),
            "torch": torch.__version__,
            "transformers": _version_of("transformers"),
            "flash-attn": _version_of("flash-attn"),
        }

    def peak_bytes(self) -> int:
        torch = self._torch
        if self.device == "cuda":
            return int(torch.cuda.max_memory_reserved(0))
        return int(torch.mps.driver_allocated_memory())

    def _release(self) -> None:
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
            self._torch.cuda.reset_peak_memory_stats(0)
        else:
            self._torch.mps.empty_cache()

    def generate(self, job, progress):
        self._release()
        progress.enter("encoding")
        peaks: dict = {}

        def on_step(state):
            if progress.stage == "encoding":
                peaks["encoding"] = self.peak_bytes()
                progress.enter("denoising", job.steps)
            progress.reached(int(state["i"]) + 1)
            if progress.step == job.steps:
                peaks["denoising"] = self.peak_bytes()
                progress.enter("decoding")

        # The package's generate() defaults sample_size to 5,292,032 samples (120 s at 44.1
        # kHz) and clamps the duration to it, so a longer request came back 120 s long
        # without a word (stable-audio-3-medium asked for 380 s on 2026-10-09). The model's
        # own config states its window (380.4 s for Medium); the package's CLI passes it.
        audio = self._model.generate(
            prompt=job.prompt,
            negative_prompt=job.negative_prompt,
            duration=float(job.duration_s),
            sample_size=int(self._model.model_config["sample_size"]),
            steps=job.steps,
            seed=job.seed,
            callback=on_step,
            disable_tqdm=True,
        )
        peaks["decoding"] = self.peak_bytes()
        samples = audio[0].to(self._torch.float32).cpu().numpy().T.copy()
        del audio
        self._release()
        made_s = samples.shape[0] / float(self.sample_rate)
        if made_s + SHORT_TOLERANCE_S < float(job.duration_s):
            raise RuntimeError(
                f"stable-audio-3 made {made_s:.1f} s of audio for a {float(job.duration_s):.1f} s "
                "request; a shorter file is not the request, so nothing is returned"
            )
        return audiocore.ArrayAudio(samples, self.sample_rate), None, peaks


ENGINES = {StableAudio3Engine.name: StableAudio3Engine}


def main() -> int:
    return audiocore.Worker(LABEL, ENGINES).serve()


if __name__ == "__main__":
    sys.exit(main())
