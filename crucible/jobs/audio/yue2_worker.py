from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import contextlib
import gc

audiocore = workerio.load_sibling("audiocore", __file__)
planning = workerio.load_sibling("planning", __file__)
scorelength = workerio.load_sibling("scorelength", __file__)

LABEL = "yue2"

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


def _ar_layer_parts(layer) -> list:
    """One DecoderLayer's AR modules: everything that is not `nar_*`."""
    return [child for name, child in layer.named_children() if not name.startswith("nar_")]


@contextlib.contextmanager
def ar_one_layer_at_a_time(model, home: "HostHomes", device):
    """While the body runs, YuE2's embeddings and each AR layer come to the card only for
    their own turn: the embeddings when they are called, layer i when its
    `input_layernorm` is (the first thing the synthesis prefill does with a layer,
    yue2/nar.py CachedNAR._prefill), and whatever was on the card before goes home first.
    So at any moment one layer of the AR half's weights is on the card, beside the prefix
    keys and values the prefill has written so far. Every forward hook is removed and the
    last resident part sent home when the body ends, and a prefill that did not walk
    every layer is refused rather than trusted."""
    backbone = model.model
    resident: list = []
    walked: list = []
    handles = []

    def arrive(parts, label):
        def hook(_module, _args):
            home.place(resident, "cpu")
            resident[:] = parts
            home.place(parts, device)
            walked.append(label)
        return hook

    try:
        handles.append(backbone.embed_tokens.register_forward_pre_hook(
            arrive([backbone.embed_tokens], "embed_tokens")))
        for index, layer in enumerate(backbone.layers):
            handles.append(layer.input_layernorm.register_forward_pre_hook(
                arrive(_ar_layer_parts(layer), index)))
        yield
    finally:
        for handle in handles:
            handle.remove()
        home.place(resident, "cpu")
    expected = ["embed_tokens", *range(len(backbone.layers))]
    if walked != expected:
        raise RuntimeError(
            f"the synthesis prefill walked {walked[:4]}...{walked[-2:]} ({len(walked)} parts); "
            f"the worker streams YuE2's AR half in the order embed_tokens, layers 0 to "
            f"{len(backbone.layers) - 1}, once each. Bring the env to its recipe "
            "(`crucible install audio`)"
        )


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
    keeps its host copy (HostHomes). With `[audio] low_vram` only what a stage uses is on
    the card and the rest waits at home. Measured on the 3090 Ti (2026-10-08): 6.37 to
    6.62 GiB over the desktop against 8.73, the audio within -105 dB of the whole model,
    about the same time.

    Under low_vram the synthesizing stage's weights on the card do not depend on the
    song's length (Victoria's 3070 laptop, 2026-10-10: a song composed to 8,960 tokens
    ran out of its 6.8 GiB cap 4 s into synthesizing). yue2-infer synthesizes a song as
    one chunk up to about 10,000 frames (`chunk_ranges`: half the context left after the
    prefix), and each chunk first runs the AR half over the whole prefix plus every codec
    token, keeping every layer's keys and values for the solve: 112 KiB a token, 1.44 GiB
    for that song's 13,483 tokens. It used to do that with the whole AR half, embeddings
    and lm_head on the card (4.03 GiB), so the keys grew on top of them and the prefill's
    last layers passed the cap. Now the synthesis prefill brings the AR half one layer at
    a time (`ar_one_layer_at_a_time`; lm_head is never used there and stays home), and the
    solve holds the NAR half (2.63 GiB) beside those keys. What still grows with the song
    is the keys and values the solve attends to, which the composing stage already holds
    for its whole token budget before it writes a token, so synthesizing stays below
    composing at any length.

    Four of yue2-infer's internals are replaced or wrapped, and each is checked first so
    another version is refused rather than half-applied: the pipeline's `_load_model`
    (which moves the whole model to the card on every stage), its `decode` (which loads
    the VAE on first use, so its homes are taken there), and, for low_vram,
    `yue2.nar._offload_ar` (which only moves AR off for the solve, leaving NAR where it
    was) and `yue2.nar.CachedNAR._prefill` (which runs with the whole AR half resident).
    Returns the host bytes the homes hold, by part, filled as each part loads."""
    import inspect

    import yue2.nar as nar_module
    from yue2.modeling_vae import YuE2VAE
    from yue2.modeling_yue2 import YuE2ForCausalLM

    expected = ["model", "enabled"]
    found = list(inspect.signature(getattr(nar_module, "_offload_ar", lambda: None)).parameters)
    decode_found = list(inspect.signature(pipe.decode).parameters) if hasattr(pipe, "decode") else None
    cached_nar = getattr(nar_module, "CachedNAR", None)
    prefill_found = (
        list(inspect.signature(cached_nar._prefill).parameters)
        if cached_nar is not None and hasattr(cached_nar, "_prefill") else None
    )
    if (
        found != expected
        or not hasattr(pipe, "_load_model")
        or decode_found != ["latents", "full", "vae"]
        or not hasattr(pipe, "_vae")
        or prefill_found != ["self"]
    ):
        raise RuntimeError(
            f"the worker is written against yue2-infer's nar._offload_ar{tuple(expected)}, "
            "nar.CachedNAR._prefill(self), YuE2Pipeline._load_model and "
            "YuE2Pipeline.decode(latents, full, vae); this yue2-infer has "
            f"_offload_ar{tuple(found)}, _prefill{tuple(prefill_found or ())} and "
            f"decode{tuple(decode_found or ())}. Bring the env to its recipe "
            "(`crucible install audio`)"
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
        if for_nar:
            # Synthesis: both halves wait at home. Its prefill brings the AR half one
            # layer at a time and its solve brings the NAR half (`swap`).
            home.place(ar, "cpu")
            torch.cuda.empty_cache()
        home.place(_adapters(model), device)
        if not for_nar:
            home.place(ar, device)
        misplaced = [
            name for name, tensor in model.named_parameters()
            if (tensor.device.type == "cpu") != (".nar_" in name or (for_nar and _is_ar(name)))
        ]
        if misplaced:
            raise RuntimeError(
                f"low_vram left {len(misplaced)} tensor(s) on the wrong side, e.g. {misplaced[:3]}"
            )
        return model

    @contextlib.contextmanager
    def swap(model, enabled):
        # yue2-infer enters this around each chunk's solve, after that chunk's prefill.
        # The AR half is already home (the prefill sent each layer back as it finished)
        # and stays there afterwards: the next chunk's prefill streams it again, and every
        # later stage places it through `load`.
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

    prefill = cached_nar._prefill

    def prefill_one_layer_at_a_time(self):
        with ar_one_layer_at_a_time(self.model, homes["model"], device):
            prefill(self)

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
        cached_nar._prefill = prefill_one_layer_at_a_time
    return held


def _adapters(model) -> list:
    """What low_vram keeps on the card in every stage: lm_head's siblings (the NAR
    adapters vae2llm, llm2vae, time_embedder and latent_pos_embed) and the backbone's
    final norm and rotary embedding."""
    return [
        *(child for name, child in model.named_children() if name not in ("model", "lm_head")),
        *(part for name, part in model.model.named_children() if name not in ("layers", "embed_tokens")),
    ]


def _is_ar(name: str) -> bool:
    """Whether a parameter of YuE2ForCausalLM is in the AR half `_halves` names: the
    embeddings, lm_head, or a layer's non-`nar_*` modules."""
    return name.startswith(("model.embed_tokens.", "lm_head.")) or (
        name.startswith("model.layers.") and ".nar_" not in name
    )


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


def planned_from(job) -> str:
    """What YuE2 plans the score from. A sung song: its lyrics. An instrumental: its
    planning lyrics - the client's, or the pool's set for its seed, which the server
    settles for every instrumental sent without `lyrics` (crucible/jobs/audio/planning.py),
    so the melody has a sung song's bounded phrases - or, when the client shaped it with
    section tags in `lyrics`, those tags. Empty sections alone bounded nothing: 4 of 13
    instrumentals ran the score to its 4096-token cap (Victoria's laptop, 2026-10-10)."""
    if job.planning_lyrics is not None:
        if not job.instrumental or job.lyrics is not None:
            raise RuntimeError(
                "planning lyrics plan an instrumental that has no lyrics; this request has "
                f"instrumental={job.instrumental} and lyrics "
                f"{'set' if job.lyrics is not None else 'unset'}, which the server refuses"
            )
        return job.planning_lyrics
    if job.lyrics is None:
        raise RuntimeError(
            "this song has neither lyrics nor planning lyrics to plan its score from; the "
            "server settles planning lyrics for every instrumental sent without lyrics"
        )
    return job.lyrics


def length_record(wanted, attempts: list, resized) -> dict:
    """A song's `audio.length`: the range asked (null ends when not sent), the nominal
    seconds of the score it was composed from, and every score planned to get there.
    `resized_planning_lyrics` is the text of a pool set grown or cut to land in the range
    (the server folds it into `audio.planning_lyrics`); null when the set was not
    resized."""
    last = attempts[-1]
    return {
        "min_duration_s": None if wanted is None else wanted.minimum,
        "max_duration_s": None if wanted is None else wanted.maximum,
        "score_seconds": last["score_seconds"],
        "in_range": last["in_range"],
        "attempts": attempts,
        "resized_planning_lyrics": resized,
    }


def range_words(wanted) -> str:
    if wanted.minimum is not None and wanted.maximum is not None:
        return f"{wanted.minimum:g}-{wanted.maximum:g} s"
    if wanted.minimum is not None:
        return f"at least {wanted.minimum:g} s"
    return f"at most {wanted.maximum:g} s"


def ratio_needed(wanted, seconds: float) -> float:
    """What the song's length must be multiplied by to reach the nearer end of the range."""
    bound = wanted.minimum if wanted.minimum is not None and seconds < wanted.minimum else wanted.maximum
    return round(bound / seconds, 3)


def out_of_range_details(wanted, attempts: list) -> dict:
    seconds = attempts[-1]["score_seconds"]
    middle = None
    if wanted.minimum is not None and wanted.maximum is not None:
        middle = round((wanted.minimum + wanted.maximum) / 2 / seconds, 3)
    return {
        **wanted.to_dict(),
        "score_seconds": seconds,
        "direction": "longer" if wanted.minimum is not None and seconds < wanted.minimum else "shorter",
        "ratio_needed": ratio_needed(wanted, seconds),
        "ratio_to_middle": middle,
        "score": attempts[-1]["score"],
        "attempts": attempts,
    }


def out_of_range_words(wanted, seconds: float) -> str:
    longer = wanted.minimum is not None and seconds < wanted.minimum
    return (
        f"the score YuE2 wrote for these words lasts {seconds:.1f} s (its bars at its tempo), "
        f"outside the {range_words(wanted)} asked, so nothing was composed. The words are "
        f"the client's and Crucible never changes them: make them "
        f"{'longer' if longer else 'shorter'}, to about {ratio_needed(wanted, seconds):g}x "
        "(more or fewer sections or lines), and send the song again. The plan is kept in "
        f"the job's {FAILED_PLAN_DIR}/"
    )


def not_reached_words(wanted, attempts: list, sizing) -> str:
    tried = "; ".join(
        f"{a['lines']} lines -> {a['score_seconds']} s" for a in attempts
    )
    return (
        f"{len(attempts)} score(s) planned for this instrumental, none inside the "
        f"{range_words(wanted)} asked ({tried}); the budget is "
        f"{planning.MAX_LENGTH_ATTEMPTS} and this pool set comes in "
        f"{len(sizing.sizes)} sizes. Nothing was composed. Send it again with another "
        "seed or planning_set, or a wider range. The plans are kept in the job's "
        f"{FAILED_PLAN_DIR}/"
    )


class YuE2Engine:
    name = "yue2"
    spans = SPANS
    notes = None
    decode_stages = None
    # The score's length against the range asked (length_record); set by every song.
    length = None
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
        lyrics that are only its section tags, so nothing is sung. The words the score was
        planned from (`planning_lyrics`) never reach this second plan."""
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
        try:
            converted, transfer = convert_score(planned.abc)
            validate_score(converted)
        except Exception as exc:
            # The score ended but the skill's checks refuse it (a bar longer than its
            # meter: the first planning-lyrics song on the PC, 2026-10-10). The plan is the
            # evidence, kept as for a truncated score; the refusal itself goes on unchanged.
            planned.save(os.path.join(os.path.dirname(job.output_path), FAILED_PLAN_DIR))
            raise RuntimeError(
                f"YuE2's score for this instrumental cannot be moved to the instrument "
                f"({type(exc).__name__}: {exc}); send it again with another seed. The plan "
                f"it wrote is kept in the job's {FAILED_PLAN_DIR}/"
            ) from exc
        cot = "full" if parse_abc(converted).voices["Vocal"].chords else "melody"
        fixed = self._pipe.plan(
            job.tags, lyric_tags(converted), abc=converted, cot=cot,
            seed=job.seed, cfg_scale=job.cfg,
        )
        if fixed.abc != converted:
            raise RuntimeError("YuE2 did not keep the instrumental score it was given")
        self.notes = {"instrumental_transfer": transfer, "planned_score": planned.abc}
        return fixed

    def _score(self, job, progress):
        """Plan the song's score, and see that it lands in the length range asked.

        Every score planned is an attempt with its structure, its nominal seconds
        (scorelength.read: its bars at its tempo, read as written) and whether it holds.
        With no range asked there is one attempt and nothing is checked. A song planned
        from words the client owns - sung lyrics, its own planning_lyrics, section-tag
        lyrics - is never altered: outside the range it is refused
        `song_length_out_of_range` before anything is composed, with the ratio the words
        need, and the worker stays loaded for the client's next try. An instrumental
        planned from the server's pool set is re-planned, the SCORE only, with the set
        grown or cut by whole sections (planning.Sizing, planning.aim) - the first size
        aimed at with the pool's measured seconds a line, every later one with this
        request's own - up to planning.MAX_LENGTH_ATTEMPTS scores, then refused
        `instrumental_length_not_reached` with every attempt's numbers. The same seed
        and params plan the same attempts, so a kept request reproduces the song.

        Returns the plan to compose from, and sets self.length (the done record's
        `audio.length`) and self.decode_stages["scoring"] (the last score's decode)."""
        pipe = self._pipe
        caps = pipe.generation_config
        stop = lambda: progress.asked_to_stop
        wanted = None
        if job.min_duration_s is not None or job.max_duration_s is not None:
            wanted = planning.LengthRange(
                None if job.min_duration_s is None else float(job.min_duration_s),
                None if job.max_duration_s is None else float(job.max_duration_s),
                float(job.longest_s),
            )
        words = planned_from(job)
        sizing = None
        count = None
        if wanted is not None and job.planning_resizable:
            sizing = planning.Sizing.of(words)
            count = planning.aim(sizing, wanted, planning.PRIOR_SECONDS_PER_LINE, set())
        attempts: list = []
        plans: list = []
        progress.enter("scoring", caps.abc.max_tokens)
        if sizing is not None and count != sizing.written:
            progress.note(
                f"the planning set is sized for the {range_words(wanted)} asked: "
                f"{sizing.lines(count)} lines ({', '.join(sizing.structure(count))})"
            )
        while True:
            text = words if sizing is None else sizing.text(count)
            ticks = audiocore.Throttled(progress, TOKENS_PER_REPORT)
            plan = pipe.plan(
                job.tags,
                text,
                seed=job.seed,
                cfg_scale=job.cfg,
                cancelled=stop,
                on_token=lambda *_: ticks.tick(),
            )
            plans.append(plan)
            # The score YuE2 decoded. An instrumental then re-plans from a fixed score,
            # which decodes nothing, so the last attempt's is the stage's decode.
            self.decode_stages["scoring"] = decode_facts(
                plan.timing, plan.truncated, caps.abc.max_tokens, self.low_vram
            )
            attempt = self._attempt(len(attempts) + 1, plan, sizing, count, wanted)
            attempts.append(attempt)
            resized = None if sizing is None or count == sizing.written else text
            self.length = length_record(wanted, attempts, resized)
            if job.instrumental and (plan.truncated or not plan.abc):
                # No melody to move to the instrument: _instrumental_plan refuses it by
                # name and keeps the plan, as for any instrumental.
                return plan
            if wanted is None or attempt["in_range"]:
                return plan
            if attempt["score_seconds"] is None:
                self._keep_plans(job, plans)
                raise audiocore.Refused(
                    "score_length_unreadable",
                    f"YuE2's score for this song cannot be measured ({attempt['unread']}), so "
                    f"it cannot be held to the {range_words(wanted)} asked; nothing was "
                    f"composed. The plan is kept in the job's {FAILED_PLAN_DIR}/",
                    {**wanted.to_dict(), "attempts": attempts},
                )
            if sizing is None:
                self._keep_plans(job, plans)
                raise audiocore.Refused(
                    "song_length_out_of_range",
                    out_of_range_words(wanted, attempt["score_seconds"]),
                    out_of_range_details(wanted, attempts),
                )
            tried = {entry["body_sections"] for entry in attempts}
            measured = sum(entry["score_seconds"] / entry["lines"] for entry in attempts) / len(attempts)
            count = None
            if len(attempts) < planning.MAX_LENGTH_ATTEMPTS:
                count = planning.aim(sizing, wanted, measured, tried)
            if count is None:
                self._keep_plans(job, plans)
                raise audiocore.Refused(
                    "instrumental_length_not_reached",
                    not_reached_words(wanted, attempts, sizing),
                    {**wanted.to_dict(), "attempts": attempts,
                     "max_attempts": planning.MAX_LENGTH_ATTEMPTS},
                )
            progress.note(
                f"score {len(attempts)} is {attempt['score_seconds']:.1f} s, outside the "
                f"{range_words(wanted)} asked; planning it again with "
                f"{sizing.lines(count)} lines ({', '.join(sizing.structure(count))})"
            )

    def _attempt(self, number, plan, sizing, count, wanted) -> dict:
        """One score's record: its structure (null for words not resized), how its decode
        ended, and its nominal length as written, or why it could not be read."""
        reading, unread = None, None
        if plan.abc:
            try:
                reading = scorelength.read(plan.abc)
            except scorelength.ScoreLengthError as exc:
                unread = str(exc)
        else:
            unread = "the model wrote no score"
        seconds = None if reading is None else round(reading.seconds, 2)
        return {
            "attempt": number,
            "body_sections": count,
            "structure": None if sizing is None else sizing.structure(count),
            "lines": None if sizing is None else sizing.lines(count),
            "score_seconds": seconds,
            "score": None if reading is None else reading.to_dict(),
            "unread": unread,
            "score_tokens": plan.timing["output_tokens"],
            "score_ended": "cap" if plan.truncated else "eos",
            "in_range": None if wanted is None or seconds is None else wanted.holds(seconds),
        }

    def _keep_plans(self, job, plans) -> None:
        """A song stopped for its length keeps every score it planned, as evidence, in the
        job's failed-plan/ (one directory, or attempt-1/, attempt-2/... for several)."""
        kept = os.path.join(os.path.dirname(job.output_path), FAILED_PLAN_DIR)
        if len(plans) == 1:
            plans[0].save(kept)
            return
        for number, plan in enumerate(plans, start=1):
            plan.save(os.path.join(kept, f"attempt-{number}"))

    def _stages(self, job, progress, peaks):
        pipe = self._pipe
        caps = pipe.generation_config
        stop = lambda: progress.asked_to_stop
        self.notes = None
        self.length = None
        self.decode_stages = {}
        plan = self._score(job, progress)
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
