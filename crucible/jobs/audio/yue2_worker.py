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

# The sections an instrumental is planned with when the job sends no lyrics: the yue2-music
# skill's own default (skills/yue2-music/instrumental/scripts/instrumental.py).
INSTRUMENTAL_SECTIONS = "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n"

# Where an instrumental whose score came back empty or truncated keeps the plan YuE2 wrote
# (SymbolicPlan.save: its tokens, timing and how it ended), in the job's own directory.
FAILED_PLAN_DIR = "failed-plan"

GIB = 1024**3

YUE2_RESERVE_GIB = 2

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


def _halves(model) -> tuple[list, list]:
    """The backbone's two halves. Each DecoderLayer holds AR modules (`self_attn`, `mlp` and
    their norms), used to write the score and the song tokens and to prefill synthesis,
    and `nar_*` modules, used only by the synthesis ODE solve (yue2/nar.py). Embeddings
    and lm_head go with AR; the small adapters (vae2llm, time_embedder, latent_pos_embed)
    stay on the card throughout."""
    ar, nar = [model.model.embed_tokens, model.lm_head], []
    for layer in model.model.layers:
        for name, child in layer.named_children():
            (nar if name.startswith("nar_") else ar).append(child)
    if not nar:
        raise RuntimeError("this YuE2 has no nar_* modules; low_vram cannot split it")
    return ar, nar


class HostHomes:
    """Every parameter and buffer of a model keeps, for the life of the worker, the one
    host copy it was loaded into. Sending a module to the card copies from that copy;
    bringing it back points the module at it again. Nothing is written to host memory
    after the load, so a worker that makes song after song holds the same host memory
    after the hundredth as after the first.

    Why (Victoria's 3070 laptop, 2026-10-10: the worker was OOM-killed at 15.5 GB of
    anonymous memory on track 12 of an album, in a 15.8 GB WSL guest): `Module.to("cpu")`
    allocates a fresh host copy of every tensor it moves, and YuE2 moves its backbone
    between the card and host memory several times a song - yue2-infer's own decode parks
    the whole 7.2 GB backbone in host memory while its VAE runs, and `[audio] low_vram`
    swaps the two halves around every synthesis chunk. Each round freed gigabytes of host
    copies into glibc's heap and allocated gigabytes more, and the heap kept the freed
    pages: measured on the 3090 Ti, two songs left 6.0 GB free-but-held in the heap
    (`malloc_trim` handed 4.3 GB of it back), while the tensors actually referenced stayed
    at 7.5 GB song after song. It was allocator retention fed by churn, not a reference
    leak, so the cure is to stop the churn rather than to trim after it.

    The model's weights never change while it serves (inference only, `eval()`), so the
    loaded copy is always the right one to come back to. When the loader maps the
    safetensors file, the homes are that file's pages: clean, file-backed, and reclaimable
    by the kernel, never anonymous memory."""

    def __init__(self, root, torch, label: str) -> None:
        self._torch = torch
        self._label = label
        self._slots: dict = {}
        storages: dict = {}
        for module in root.modules():
            slots = []
            for is_parameter, table in ((True, module._parameters), (False, module._buffers)):
                for name, tensor in table.items():
                    if tensor is None:
                        continue
                    if tensor.device.type != "cpu":
                        raise RuntimeError(
                            f"{label}'s {name!r} is on {tensor.device} as its host home is "
                            "taken; the home is the copy the load put in host memory"
                        )
                    home = tensor.detach()
                    storage = home.untyped_storage()
                    storages[storage.data_ptr()] = storage.nbytes()
                    slots.append((table, name, is_parameter, home))
            self._slots[module] = slots
        self.bytes = sum(storages.values())
        homes = self

        def to(*args, **kwargs):
            if kwargs or len(args) != 1 or not isinstance(args[0], (str, torch.device)):
                raise RuntimeError(
                    f"{label} moves only between the card and its host home, given one "
                    f"device; this call passed {args!r} {kwargs!r}"
                )
            homes.place([root], args[0])
            return root

        # yue2-infer's pipeline moves the model and its decoder with `.to(...)` (decode
        # parks the backbone, sends the VAE to the card and brings it back); routing that
        # through the homes is what keeps those moves from allocating host copies too.
        root.to = to

    def _device(self, device):
        target = self._torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = self._torch.device("cuda", self._torch.cuda.current_device())
        return target

    def place(self, modules, device) -> None:
        """Put every parameter and buffer of `modules` on `device`: a copy of its home on
        the card, the home itself in host memory."""
        target = self._device(device)
        with self._torch.no_grad():
            for module in modules:
                for part in module.modules():
                    slots = self._slots.get(part)
                    if slots is None:
                        raise RuntimeError(
                            f"{type(part).__name__} in {self._label} was not part of it when "
                            "its host homes were taken, so it has no home to come back to"
                        )
                    for table, name, is_parameter, home in slots:
                        current = table[name]
                        if current.device == target:
                            continue
                        moved = home if target.type == "cpu" else home.to(target)
                        if is_parameter:
                            current.data = moved
                        else:
                            table[name] = moved


def own_residency(pipe, torch, low_vram: bool) -> dict:
    """The worker decides where each part of YuE2 lives, on a CUDA card, and every part
    keeps its host copy (HostHomes). With `[audio] low_vram` only the half a stage uses is
    on the card and the other waits at home. Measured on the 3090 Ti (2026-10-08): 6.37 to
    6.62 GiB over the desktop against 8.73, the audio within -105 dB of the whole model,
    about the same time.

    Three of yue2-infer's internals are replaced or wrapped, and each is checked first so
    another version is refused rather than half-applied: the pipeline's `_load_model`
    (which moves the whole model to the card on every stage), its `decode` (which loads
    the VAE on first use, so its homes are taken there), and, for low_vram,
    `yue2.nar._offload_ar` (which only moves AR off for the solve, leaving NAR where it
    was). Returns the host bytes the homes hold, by part, filled as each part loads."""
    import contextlib
    import inspect

    import yue2.nar as nar_module
    from yue2.modeling_vae import YuE2VAE
    from yue2.modeling_yue2 import YuE2ForCausalLM

    expected = ["model", "enabled"]
    found = list(inspect.signature(getattr(nar_module, "_offload_ar", lambda: None)).parameters)
    decode_found = list(inspect.signature(pipe.decode).parameters) if hasattr(pipe, "decode") else None
    if (
        found != expected
        or not hasattr(pipe, "_load_model")
        or decode_found != ["latents", "full", "vae"]
        or not hasattr(pipe, "_vae")
    ):
        raise RuntimeError(
            f"the worker is written against yue2-infer's nar._offload_ar{tuple(expected)}, "
            "YuE2Pipeline._load_model and YuE2Pipeline.decode(latents, full, vae); this "
            f"yue2-infer has _offload_ar{tuple(found)} and decode{tuple(decode_found or ())}. "
            "Bring the env to its recipe (`crucible install audio`)"
        )
    if pipe.quantization != "none" or pipe.backend == "vllm":
        raise RuntimeError(
            f"the worker holds YuE2 unquantized on the torch backend; this pipeline is "
            f"{pipe.quantization} on {pipe.backend}"
        )
    device = pipe.device
    held: dict = {}
    homes: dict = {}

    def load(for_nar=False):
        if pipe._model is None:
            pipe._model = YuE2ForCausalLM.from_pretrained(
                pipe.model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).eval()
            homes["model"] = HostHomes(pipe._model, torch, "YuE2's backbone")
            held["model"] = homes["model"].bytes
        model = pipe._model
        home = homes["model"]
        if not low_vram:
            home.place([model], device)
            return model
        ar, nar = _halves(model)
        home.place(nar, "cpu")
        home.place([child for name, child in model.named_children() if name != "model"], device)
        home.place([part for name, part in model.model.named_children() if name != "layers"], device)
        home.place(ar, device)
        misplaced = [
            name for name, tensor in model.named_parameters()
            if (".nar_" in name) != (tensor.device.type == "cpu")
        ]
        if misplaced:
            raise RuntimeError(
                f"low_vram left {len(misplaced)} tensor(s) on the wrong side, e.g. {misplaced[:3]}"
            )
        return model

    @contextlib.contextmanager
    def swap(model, enabled):
        if not enabled:
            yield
            return
        ar, nar = _halves(model)
        home = homes["model"]
        home.place(ar, "cpu")
        torch.cuda.empty_cache()
        home.place(nar, device)
        try:
            yield
        finally:
            home.place(nar, "cpu")
            torch.cuda.empty_cache()
            home.place(ar, device)

    decode = pipe.decode

    def decode_at_home(latents, *, full=False, vae=None):
        if vae is not None:
            raise RuntimeError("the worker decodes with the pipeline's own VAE, never another")
        if pipe._vae is None:
            # The same load yue2-infer's decode makes on first use, taken here so its
            # homes exist before decode sends it to the card.
            pipe._vae = YuE2VAE.from_pretrained(
                pipe.vae_dir, decoder_only=True, device="cpu", local_files_only=True
            )
            homes["vae"] = HostHomes(pipe._vae, torch, "YuE2's VAE")
            held["vae"] = homes["vae"].bytes
        return decode(latents, full=full)

    pipe._load_model = load
    pipe.decode = decode_at_home
    if low_vram:
        pipe.offload_ar = True
        nar_module._offload_ar = swap
    return held


def decode_facts(timing: dict, truncated: bool, cap: int, low_vram: bool) -> dict:
    """What one autoregressive stage did, from yue2-infer's own account of it
    (`yue2.sampling.generate_tokens`' timing and its `truncated`), so a finished job says
    how each stage ended without anyone re-running the seed. A stage is one pass over one
    prefix: YuE2 decodes neither the score nor the song tokens in segments.

    `ended` is "eos" when the model wrote its end token and "cap" when the stage ran to
    `cap` tokens without one - a runaway, the 6-minute song on Victoria's 3070
    (2026-10-09). `tokens` counts every token the model wrote, the end token included."""
    return {
        "tokens": timing["output_tokens"],
        "cap": cap,
        "ended": "cap" if truncated else "eos",
        "execution": timing["execution"],
        "attention": timing["attention"],
        "low_vram": low_vram,
        "prefix_tokens": timing["prefix_tokens"],
        "cfg_branches": timing["cfg_branches"],
        "seconds": round(timing["seconds"], 2),
        "prefill_seconds": round(timing["prefill_seconds"], 2),
        "tokens_per_second": round(timing["output_tps"], 1),
    }


def _version_of(distribution: str):
    from importlib import metadata

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


class YuE2Engine:
    name = "yue2"
    spans = SPANS
    notes = None
    decode_stages = None
    # The host bytes each part's HostHomes hold, by part ("model", "vae"), filled as each
    # loads; None on Apple silicon, where yue2-infer's own moves stay.
    host_homes = None

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
        low_vram = workerio.require(request, "low_vram", bool, LABEL, audiocore.WHY_REQUIRED)
        if low_vram and self.device != "cuda":
            raise RuntimeError(
                f"low_vram holds half of YuE2 on a CUDA card at a time; this worker runs on "
                f"{self.device}, where no manifest offers it"
            )
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
        # On Apple silicon the card and host memory are the same memory, so a host home
        # beside the copy on the GPU would hold the model twice; yue2-infer's own moves stay.
        self.host_homes = own_residency(self._pipe, torch, low_vram) if self.device == "cuda" else None
        if low_vram:
            # YuE2 capped this process at the card less 2 GiB, which on an 8 GiB card is
            # below the 6.05 GiB a halved composing stage reserves. Crucible admitted the
            # load against its own measured need, so that need is the cap.
            workerio.cap_memory(
                torch,
                workerio.require(request, "memory_cap_bytes", int, LABEL, audiocore.WHY_REQUIRED),
            )
        self.low_vram = low_vram
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

    def _instrumental_plan(self, job, planned):
        """YuE2's instrumental workflow (the yue2-music skill, vendored in yue2music/): the
        score YuE2 just wrote has its vocal melody moved, note for note, into the
        instrumental voice - the skill's own checks refuse any change of pitch, onset,
        duration, meter, tempo or harmony - and YuE2 then renders THAT fixed score with
        lyrics that are only its section tags, so nothing is sung."""
        tools = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yue2music")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        from abc_tools import parse_abc
        from instrumental import lyric_tags, validate_score
        from instrumentalize import convert_score

        if planned.truncated or not planned.abc:
            # What the model actually wrote is the evidence: the whole plan (its tokens,
            # timing and how it ended) is saved beside the job's kept request, and the
            # error says which of the two failures it was. Victoria's two-album run had 2
            # of ~11 instrumentals end here (2026-10-10) and nothing said why.
            kept = os.path.join(os.path.dirname(job.output_path), FAILED_PLAN_DIR)
            planned.save(kept)
            tokens = planned.timing["output_tokens"]
            if planned.truncated:
                how = (f"truncated: it ran to its {self._pipe.generation_config.abc.max_tokens}"
                       "-token cap without ending")
            else:
                how = f"empty: the model ended it after {tokens} tokens with no score in them"
            raise RuntimeError(
                f"YuE2's score for this instrumental came back {how}, so there is no "
                "melody to move to the instrument; send it again with another seed. The "
                f"plan it wrote is kept in the job's {FAILED_PLAN_DIR}/"
            )
        converted, transfer = convert_score(planned.abc)
        validate_score(converted)
        cot = "full" if parse_abc(converted).voices["Vocal"].chords else "melody"
        fixed = self._pipe.plan(
            job.tags, lyric_tags(converted), abc=converted, cot=cot,
            seed=job.seed, cfg_scale=job.cfg,
        )
        if fixed.abc != converted:
            raise RuntimeError("YuE2 did not keep the instrumental score it was given")
        self.notes = {"instrumental_transfer": transfer, "planned_score": planned.abc}
        return fixed

    def _stages(self, job, progress, peaks):
        pipe = self._pipe
        caps = pipe.generation_config
        stop = lambda: progress.asked_to_stop
        self.notes = None
        self.decode_stages = {}
        progress.enter("scoring", caps.abc.max_tokens)
        ticks = audiocore.Throttled(progress, TOKENS_PER_REPORT)
        plan = pipe.plan(
            job.tags,
            job.lyrics if job.lyrics is not None else INSTRUMENTAL_SECTIONS,
            seed=job.seed,
            cfg_scale=job.cfg,
            cancelled=stop,
            on_token=lambda *_: ticks.tick(),
        )
        # The score YuE2 decoded. An instrumental then re-plans from a fixed score, which
        # decodes nothing, so this is the scoring stage's one decode either way.
        self.decode_stages["scoring"] = decode_facts(
            plan.timing, plan.truncated, caps.abc.max_tokens, self.low_vram
        )
        if job.instrumental:
            plan = self._instrumental_plan(job, plan)
        self._close_stage(peaks, "scoring")
        progress.enter("composing", caps.semantic.max_tokens)
        ticks = audiocore.Throttled(progress, TOKENS_PER_REPORT)
        semantic = pipe.generate_semantic(plan, cancelled=stop, on_token=lambda *_: ticks.tick())
        self.decode_stages["composing"] = decode_facts(
            semantic.timing, semantic.truncated, caps.semantic.max_tokens, self.low_vram
        )
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
