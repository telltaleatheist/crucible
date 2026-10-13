from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import workerio

sys.path.pop(0)
workerio.claim_stdout()

import contextlib
import gc
import inspect

audiocore = workerio.load_sibling("audiocore", __file__)
planning = workerio.load_sibling("planning", __file__)
scorelength = workerio.load_sibling("scorelength", __file__)
scoreheader = workerio.load_sibling("scoreheader", __file__)

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


@contextlib.contextmanager
def written_negative(plan, negative: dict, tokenizer):
    """While the body runs, the composing pass's CFG negative branch is the one the client
    wrote (docs/AUDIO.md "The negative branch"): the prefix of the song's own SongRequest -
    the same score (the plan's exact abc ids), cot, seed, cfg and id - with `negative`'s
    tags as its style and its lyrics as its lyrics. Yields that prefix.

    yue2-infer builds the negative inside `YuE2Pipeline.generate_semantic`, from
    `yue2.protocol.negative_prefix` (the instruction and the score, no tags, no lyrics), and
    takes no negative from its caller. Its own check that the plan's prefix is the request's
    and everything after it are what make a song, so rather than re-write that method here
    the one name it calls is replaced, for this one call, in `yue2.pipeline` only - the
    installed package is never edited - and put back however the body ends. The
    replacement answers only for this plan's request and abc ids, and the body must call
    it exactly once; a yue2-infer that builds its negative some other way is refused
    before a token is written."""
    import dataclasses

    import yue2.pipeline as pipeline
    from yue2 import protocol

    stock = protocol.negative_prefix
    if (
        getattr(pipeline, "negative_prefix", None) is not stock
        or "negative_prefix" not in pipeline.YuE2Pipeline.generate_semantic.__code__.co_names
    ):
        raise RuntimeError(
            "the worker writes a song's CFG negative branch through the negative_prefix "
            "yue2-infer's YuE2Pipeline.generate_semantic calls (yue2.pipeline.negative_prefix, "
            "which is yue2.protocol.negative_prefix); this yue2-infer builds its negative "
            "some other way. Bring the env to its recipe (`crucible install audio`)"
        )
    request = dataclasses.replace(plan.request, style=negative["tags"], lyrics=negative["lyrics"])
    prefix = protocol.token_prefixes(request, tokenizer, plan.abc_ids)
    calls: list = []

    def negative_prefix(positive, _tokenizer, abc_ids=None):
        if positive is not plan.request or abc_ids is None or list(abc_ids) != list(plan.abc_ids):
            raise RuntimeError(
                "the written negative branch was asked for by a request or score other than "
                "the plan it was built from; both branches compose from the plan's own abc ids"
            )
        calls.append(len(prefix))
        return list(prefix)

    pipeline.negative_prefix = negative_prefix
    try:
        yield prefix
    finally:
        pipeline.negative_prefix = stock
    if len(calls) != 1:
        raise RuntimeError(
            f"yue2-infer asked for the negative branch {len(calls)} times composing one song; "
            "the worker writes it for exactly one composing pass"
        )


class HeaderForcer:
    """Writes a planned score's fixed header lines (scoreheader.lines: {line index: text})
    into the one scoring pass, token by token, and leaves every other token to YuE2.

    It sits in front of yue2-infer's own `distribution` (forced_header) and so sees each
    step's logits before the token is drawn, with every token drawn so far (`history`). It
    follows the score line by line; when a fixed line is about to begin, the line and its
    newline - the tokenizer's own encoding of them - become the only tokens the step may
    draw, one a step, so the model's cache holds them exactly as if it had written them
    and it continues from there. YuE2 begins every header line with its field letter at a
    token boundary (the tokenizer never joins a newline to the letter after it), so a fixed
    line always starts where the model would start that line itself.

    Every other step goes to the stock `distribution` unchanged. A forced step goes to it
    too, with every logit but the forced token's at -inf, so the stock sampling (the ABC
    vocabulary, the end-token floor, the penalty, temperature, top-k, top-p and the seeded
    draw) runs as ever and can only draw that token.

    `written` lists the line indexes it wrote; `missed` the ones the score passed without
    that line beginning at a token boundary (never seen in YuE2's dialect); a fixed line
    the score ended before is in neither."""

    def __init__(self, lines: dict, tokenizer, torch, end_of_text: int) -> None:
        self._pending = {}
        for index, line in sorted(lines.items()):
            ids = list(tokenizer.encode(line + "\n"))
            if tokenizer.decode(ids) != line + "\n" or not all(0 <= t < end_of_text for t in ids):
                raise RuntimeError(
                    f"YuE2's tokenizer does not encode the header line {line!r} as ordinary "
                    "score text; the scoring pass draws only from that vocabulary"
                )
            self._pending[index] = ids
        self._tokenizer = tokenizer
        self._torch = torch
        self._queue: list = []
        self._forced = None
        self._seen = 0
        self._line = 0
        self._begun = False
        self.calls = 0
        self.written: list = []
        self.missed: list = []

    def _follow(self, history) -> None:
        if self._forced is not None:
            if not history or history[-1] != self._forced:
                raise RuntimeError(
                    "the scoring pass drew a token other than the header token it was "
                    "limited to; yue2-infer's sampling no longer draws only from what "
                    "its distribution leaves"
                )
            self._forced = None
        for token in history[self._seen:]:
            for character in self._tokenizer.decode([token]):
                if character == "\n":
                    self._line += 1
                    self._begun = False
                else:
                    self._begun = True
        self._seen = len(history)
        for index in [i for i in self._pending if i < self._line or (i == self._line and self._begun)]:
            del self._pending[index]
            self.missed.append(index)

    def __call__(self, stock, logits, sampling, history, step, phase, legacy_off=False):
        self.calls += 1
        if phase != "abc":
            raise RuntimeError(f"the header is fixed in the scoring pass, not in {phase!r}")
        self._follow(history)
        if not self._queue and not self._begun and self._line in self._pending:
            self._queue = self._pending.pop(self._line)
            self.written.append(self._line)
        if not self._queue:
            return stock(logits, sampling, history, step, phase, legacy_off)
        token = self._queue.pop(0)
        self._forced = token
        only = self._torch.full_like(logits, float("-inf"))
        only[..., token] = logits[..., token]
        return stock(only, sampling, history, step, phase, legacy_off)


@contextlib.contextmanager
def forced_header(pipe, lines: dict):
    """While the body runs, YuE2's scoring pass writes the header lines `lines` fixes
    (HeaderForcer) and plans the rest of the score itself: still one pass over one prefix,
    one prefill, one decode - forcing is not re-planning. Yields the forcer.

    yue2-infer's `generate_tokens` draws every token through the module-level
    `distribution` of `yue2.sampling` and takes no logits processor from its caller, so -
    as written_negative does for the negative branch - that one name is replaced, for this
    one call, in `yue2.sampling` only, and put back however the body ends. The installed
    package is never edited, and `pipe.plan` builds the SymbolicPlan from the tokens drawn
    exactly as it does for any score, so its prefix is token_prefixes(request, tokenizer,
    abc_ids) and generate_semantic's check holds. A yue2-infer that draws its tokens some
    other way (or the vLLM backend, which samples in its own process) is refused before a
    token is written, and one whose pass never asked is refused after it."""
    import yue2.sampling as sampling
    from yue2.protocol import EOD

    stock = sampling.distribution
    # generate_tokens is wrapped (torch.no_grad, functools.wraps), so the wrapper's own code
    # names nothing of the body's; the body is what draws through `distribution`. Read the
    # wrapper's code instead and every real yue2-infer is refused (Mac and PC, 2026-10-13).
    drawer = inspect.unwrap(sampling.generate_tokens)
    if pipe.backend == "vllm" or "distribution" not in drawer.__code__.co_names:
        raise RuntimeError(
            "the worker fixes a score's header through the distribution yue2-infer's "
            "yue2.sampling.generate_tokens draws each token from, on the torch backend; this "
            f"yue2-infer (backend {pipe.backend!r}) draws its tokens some other way. Bring "
            "the env to its recipe (`crucible install audio`)"
        )
    forcer = HeaderForcer(lines, pipe.tokenizer, sampling.torch, EOD)

    def distribution(logits, sampling_config, history, step, phase, legacy_off=False):
        return forcer(stock, logits, sampling_config, history, step, phase, legacy_off)

    sampling.distribution = distribution
    try:
        yield forcer
    finally:
        sampling.distribution = stock
    if forcer.calls == 0:
        raise RuntimeError(
            "YuE2 planned the score without drawing a token through yue2.sampling.distribution, "
            "so the header was not fixed; bring the env to its recipe (`crucible install audio`)"
        )


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


def length_record(wanted, sizing, count, plan, resized) -> dict:
    """A YuE2 song's `audio.length`: the range asked (null ends when not sent), whether
    the range shaped the plan (`sized`: only a pool-planned instrumental's set is sized),
    the structure it was sized to, and the nominal length of the one score the song was
    composed from - its bars at its tempo (scorelength.read) - with whether it landed in
    the range. A range never refuses or fails a song: it is a nudge for a pool set and a
    record for everything else. `resized_planning_lyrics` is the text of a pool set grown or
    cut by the sizing (the server folds it into `audio.planning_lyrics`); null when the set
    was planned from as written."""
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
        "min_duration_s": None if wanted is None else wanted.minimum,
        "max_duration_s": None if wanted is None else wanted.maximum,
        "sized": sizing is not None,
        "body_sections": count,
        "structure": None if sizing is None else sizing.structure(count),
        "lines": None if sizing is None else sizing.lines(count),
        "score_seconds": seconds,
        "score": None if reading is None else reading.to_dict(),
        "unread": unread,
        "in_range": None if wanted is None or seconds is None else wanted.holds(seconds),
        "resized_planning_lyrics": resized,
    }


def range_words(wanted) -> str:
    if wanted.minimum is not None and wanted.maximum is not None:
        return f"{wanted.minimum:g}-{wanted.maximum:g} s"
    if wanted.minimum is not None:
        return f"at least {wanted.minimum:g} s"
    return f"at most {wanted.maximum:g} s"


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
    # A cover's score in YuE2's tokens (_given_plan); None for a song YuE2 planned.
    score_tokens = None
    # The composing pass's CFG negative branch in tokens, stock or written by the client
    # (_compose); None when cfg 1 ran no negative branch.
    negative_prefix_tokens = None
    # The key, meter and tempo the rendered score states (scoreheader.read), set by every
    # song; the fields the client fixed are the server's record (docs/AUDIO.md "Key, meter
    # and tempo").
    score_header = None

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
        self.notes = {"instrumental_transfer": transfer, "score_before_transfer": planned.abc}
        return fixed

    def _given_plan(self, job, progress):
        """A cover: the score the client gave, which the server has checked with the skill
        and, for an instrumental, moved to the instrument (crucible/jobs/audio/cover.py),
        becomes the plan as it is - YuE2 decodes no score, so there is no scoring stage to
        report. One pass: nothing here re-plans or retries.

        YuE2 plans a score within 4096 tokens, and yue2-infer does not hold a given score
        to that budget; the skill's generate() refuses one past it before rendering
        (instrumental.py, `len(pipe.tokenizer.encode(abc)) > 4096`), and so does this, by
        the same count with the model's own tokenizer, as a refusal that leaves the
        worker loaded (audiocore.Refused)."""
        given = job.cover
        budget = self._pipe.generation_config.abc.max_tokens
        tokens = len(self._pipe.tokenizer.encode(given["abc"]))
        if tokens > budget:
            raise audiocore.Refused(
                "score_too_many_tokens",
                f"the score is {tokens} tokens in YuE2's tokenizer, past the {budget}-token "
                "budget YuE2 plans a score within; send a shorter or more compact score",
            )
        progress.enter("scoring")
        plan = self._pipe.plan(
            job.tags, given["lyrics"], abc=given["abc"], cot=given["cot"],
            seed=job.seed, cfg_scale=job.cfg,
        )
        if plan.abc != given["abc"]:
            raise RuntimeError("YuE2 did not keep the score it was given")
        self.score_tokens = tokens
        self.length = length_record(None, None, None, plan, None)
        return plan

    def _score(self, job, progress):
        """Plan the song's one score. Always one score and one render (Owen, 2026-10-10:
        "strictly one pass"); a length range never re-plans, refuses or fails a song.

        A range is a nudge for one kind of song only: an instrumental planned from the
        server's pool set (the seed's or `planning_set`'s), whose set is grown or cut by
        whole sections toward the range before its one score is planned (planning.Sizing,
        planning.aim at the pool's measured seconds a line). Words the client owns - sung
        lyrics, its own planning_lyrics, section-tag lyrics - are planned from as sent, and
        the range is only recorded against what the score measured (a `note` says so).

        A key, meter or tempo the client fixed (job.header) is written into this same one
        pass as its header line begins (forced_header); the rest of the score is YuE2's.

        Returns the plan to compose from, and sets self.length (the done record's
        `audio.length`) and self.decode_stages["scoring"]."""
        pipe = self._pipe
        caps = pipe.generation_config
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
            count = planning.aim(sizing, wanted, planning.PRIOR_SECONDS_PER_LINE)
        text = words if sizing is None else sizing.text(count)
        progress.enter("scoring", caps.abc.max_tokens)
        if sizing is not None:
            progress.note(
                f"the planning set is sized toward the {range_words(wanted)} asked: "
                f"{sizing.lines(count)} lines ({', '.join(sizing.structure(count))})"
                + ("" if count != sizing.written else ", as written")
            )
        elif wanted is not None:
            progress.note(
                f"the {range_words(wanted)} asked is recorded against the score; it does not "
                "change a song planned from the client's own words"
            )
        ticks = audiocore.Throttled(progress, TOKENS_PER_REPORT)
        fixed = scoreheader.lines(job.header)

        def plan_score():
            return pipe.plan(
                job.tags,
                text,
                seed=job.seed,
                cfg_scale=job.cfg,
                cancelled=lambda: progress.asked_to_stop,
                on_token=lambda *_: ticks.tick(),
            )

        if not fixed:
            plan = plan_score()
        else:
            progress.note(f"the score's header is fixed: {', '.join(fixed.values())}")
            with forced_header(pipe, fixed) as forcer:
                plan = plan_score()
            unwritten = [fixed[index] for index in sorted(fixed) if index not in forcer.written]
            if unwritten:
                # Recorded, never retried: the score's own header (audio.header.score)
                # says what it states instead.
                progress.note(
                    f"YuE2's score did not reach a header line where {', '.join(unwritten)} "
                    "is written (it ended, ran to its cap, or laid its header out "
                    "differently); audio.header.score says what the score states"
                )
        # The score YuE2 decoded. An instrumental then re-plans from a fixed score, which
        # decodes nothing, so this is the scoring stage's one decode either way.
        self.decode_stages["scoring"] = decode_facts(
            plan.timing, plan.truncated, caps.abc.max_tokens, self.low_vram
        )
        resized = None if sizing is None or count == sizing.written else text
        self.length = length_record(wanted, sizing, count, plan, resized)
        return plan

    def _compose(self, job, plan, stop, on_token):
        """The song's codec tokens from its one plan - planned, moved to the instrument, or
        a cover's given score alike - under the CFG negative branch the request asks for:
        YuE2's stock one (its instruction and the score), or the one the client wrote as
        `negative_tags` and `negative_lyrics` (written_negative). The positive branch is
        the plan's own either way. Every plan is made with `cfg_scale=job.cfg`, so job.cfg
        is the guidance YuE2 composes with. Sets self.negative_prefix_tokens."""
        pipe = self._pipe
        if job.cfg == 1:
            if job.negative is not None:
                # The server refuses negative words without cfg above 1; arriving here
                # they would be ignored, so they are refused rather than dropped.
                raise RuntimeError(
                    "a written negative branch arrived with cfg 1, where YuE2 runs no "
                    "negative branch; the server refuses negative_tags and negative_lyrics "
                    "without cfg above 1"
                )
            self.negative_prefix_tokens = None
            return pipe.generate_semantic(plan, cancelled=stop, on_token=on_token)
        from yue2 import protocol

        if plan.request.guidance != job.cfg:
            raise RuntimeError(
                f"the plan composes with guidance {plan.request.guidance:g} and the request "
                f"asked cfg {job.cfg:g}; every plan is made with the request's cfg"
            )
        if job.negative is None:
            self.negative_prefix_tokens = len(
                protocol.negative_prefix(plan.request, pipe.tokenizer, plan.abc_ids)
            )
            return pipe.generate_semantic(plan, cancelled=stop, on_token=on_token)
        with written_negative(plan, job.negative, pipe.tokenizer) as prefix:
            # yue2-infer refuses a negative that leaves the song no room in its context
            # (sampling.generate_tokens); said here by name before the pass, and the
            # worker stays loaded.
            room = pipe.generation_config.context - pipe.generation_config.semantic.max_tokens
            if len(prefix) > room:
                raise audiocore.Refused(
                    "negative_too_long",
                    f"the written negative branch is {len(prefix)} tokens in YuE2's tokenizer, "
                    f"and YuE2 composes a song of up to "
                    f"{pipe.generation_config.semantic.max_tokens} tokens after it within its "
                    f"{pipe.generation_config.context}-token context, which leaves {room}; send "
                    "shorter negative_tags or negative_lyrics",
                )
            semantic = pipe.generate_semantic(plan, cancelled=stop, on_token=on_token)
        self.negative_prefix_tokens = len(prefix)
        return semantic

    def _stages(self, job, progress, peaks):
        pipe = self._pipe
        caps = pipe.generation_config
        stop = lambda: progress.asked_to_stop
        self.notes = None
        self.length = None
        self.decode_stages = {}
        self.score_tokens = None
        self.negative_prefix_tokens = None
        self.score_header = None
        if job.cover is not None:
            if job.header is not None:
                raise RuntimeError(
                    "a cover renders the score it is given, whose header is its own; the "
                    "server refuses key, meter and bpm beside a score"
                )
            plan = self._given_plan(job, progress)
        else:
            plan = self._score(job, progress)
            if job.instrumental:
                plan = self._instrumental_plan(job, plan)
        self.score_header = scoreheader.read(plan.abc)
        self._close_stage(peaks, "scoring")
        progress.enter("composing", caps.semantic.max_tokens)
        ticks = audiocore.Throttled(progress, TOKENS_PER_REPORT)
        semantic = self._compose(job, plan, stop, lambda *_: ticks.tick())
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
