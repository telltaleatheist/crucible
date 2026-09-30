from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from crucible.jobs.image import inpaint

WORKER = Path(__file__).resolve().parents[1] / "crucible" / "jobs" / "image" / "worker.py"


def _save(path: Path, picture: Image.Image, **options) -> Path:
    picture.save(path, **options)
    return path


def _mask(width: int, height: int, box: tuple[int, int, int, int]) -> Image.Image:
    drawn = Image.new("L", (width, height), 0)
    drawn.paste(255, box)
    return drawn


@pytest.mark.parametrize(
    ("suffix", "options"),
    [
        (".png", {"format": "PNG"}),
        (".jpg", {"format": "JPEG", "quality": 90}),
        (".jpg", {"format": "JPEG", "progressive": True}),
        (".webp", {"format": "WEBP", "lossless": True}),
        (".webp", {"format": "WEBP", "quality": 80}),
    ],
)
def test_picture_size_reads_the_header_of_each_format(tmp_path: Path, suffix: str, options: dict) -> None:
    path = _save(tmp_path / f"picture{suffix}", Image.new("RGB", (333, 217), (10, 20, 30)), **options)
    assert inpaint.picture_size(path) == (333, 217)


def test_picture_size_reads_an_extended_webp_and_says_none_for_anything_else(tmp_path: Path) -> None:
    with_alpha = _save(tmp_path / "alpha.webp", Image.new("RGBA", (300, 200), (1, 2, 3, 128)), format="WEBP")
    assert inpaint.picture_size(with_alpha) == (300, 200)
    other = tmp_path / "notes.txt"
    other.write_bytes(b"hello, this is not a picture at all")
    assert inpaint.picture_size(other) is None


def test_the_region_is_white_at_the_job_size_and_black_is_kept() -> None:
    drawn = _mask(64, 32, (0, 0, 32, 32))
    region = inpaint.region_of(drawn, 128, 64)
    assert region.shape == (64, 128) and region.dtype == np.float32
    assert region[:, :60].min() == 1.0 and region[:, 68:].max() == 0.0
    grey = Image.new("L", (32, 32), 127)
    assert not inpaint.region_of(grey, 32, 32).any()
    assert inpaint.region_of(Image.new("RGB", (32, 32), (255, 255, 255)), 32, 32).all()


def test_the_feather_is_soft_inside_the_edge_and_never_touches_a_kept_pixel() -> None:
    region = np.zeros((64, 64), dtype=np.float32)
    region[:, 32:] = 1.0
    hard = inpaint.feathered(region, 0)
    assert np.array_equal(hard, region)
    soft = inpaint.feathered(region, 8)
    assert soft[:, :32].max() == 0.0
    assert soft[:, 48:].min() > 0.99
    row = soft[32, 32:48]
    assert row[0] < 0.2 and np.all(np.diff(row) >= 0)


def test_the_latent_grid_takes_any_pixel_of_a_tile() -> None:
    region = np.zeros((64, 96), dtype=np.float32)
    region[17, 40] = 1.0
    grid = inpaint.latent_grid(region)
    assert grid.shape == (4, 6)
    assert grid.sum() == 1.0 and grid[1, 2] == 1.0
    with pytest.raises(ValueError):
        inpaint.latent_grid(np.zeros((40, 64), dtype=np.float32))


def test_packing_matches_both_engines_latent_layouts() -> None:
    height, width, channels = 3, 5, 4
    grid = np.arange(height * width, dtype=np.float32).reshape(height, width)
    ours = inpaint.packed(grid)
    assert ours.shape == (1, height * width, 1)
    latents = np.random.default_rng(0).normal(size=(1, channels, 1, height, width)).astype(np.float32)
    latents[:, 0, 0] = grid
    diffusers = latents.reshape(1, channels, height * width).transpose(0, 2, 1)
    mflux = latents[:, :, 0].transpose(0, 2, 3, 1).reshape(1, height * width, channels)
    assert np.array_equal(diffusers[:, :, :1], ours)
    assert np.array_equal(mflux[:, :, :1], ours)


def test_the_blend_keeps_the_model_inside_and_puts_the_input_back_outside() -> None:
    rng = np.random.default_rng(1)
    latents, clean, noise = (rng.normal(size=(1, 6, 4)).astype(np.float32) for _ in range(3))
    mask = inpaint.packed(np.array([[1, 1, 0], [0, 1, 0]], dtype=np.float32))
    last = inpaint.blend_step(latents, clean, noise, mask, 0.0)
    inside, outside = mask[0, :, 0] == 1, mask[0, :, 0] == 0
    assert np.array_equal(last[0, inside], latents[0, inside])
    assert np.array_equal(last[0, outside], clean[0, outside])
    first = inpaint.blend_step(latents, clean, noise, mask, 1.0)
    assert np.array_equal(first[0, outside], noise[0, outside])
    middle = inpaint.blend_step(latents, clean, noise, mask, 0.25)
    assert np.allclose(middle[0, outside], 0.75 * clean[0, outside] + 0.25 * noise[0, outside])


def test_the_paste_back_keeps_every_pixel_outside_the_mask_exactly() -> None:
    rng = np.random.default_rng(2)
    original = Image.fromarray(rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8), mode="RGB")
    generated = Image.new("RGB", (96, 64), (0, 0, 255))
    region = np.zeros((64, 96), dtype=np.float32)
    region[16:48, 32:80] = 1.0
    out = np.asarray(inpaint.paste_back(generated, original, inpaint.feathered(region, 6)))
    kept = np.asarray(original)
    assert np.array_equal(out[region == 0], kept[region == 0])
    assert np.array_equal(out[28:36, 50:62], np.broadcast_to([0, 0, 255], (8, 12, 3)))


def test_load_mask_refuses_a_mask_of_another_size_and_one_that_selects_nothing(tmp_path: Path) -> None:
    image = _save(tmp_path / "image.png", Image.new("RGB", (256, 256), (200, 0, 0)))
    small = _save(tmp_path / "small.png", _mask(128, 128, (0, 0, 64, 64)))
    black = _save(tmp_path / "black.png", Image.new("L", (256, 256), 0))
    with pytest.raises(inpaint.MaskRefused) as refused:
        inpaint.load_mask(str(small), str(image), 256, 256, 8)
    assert refused.value.code == "mask_size_mismatch" and "128x128" in str(refused.value)
    with pytest.raises(inpaint.MaskRefused) as refused:
        inpaint.load_mask(str(black), str(image), 256, 256, 8)
    assert refused.value.code == "mask_empty"


def test_load_mask_builds_the_region_feather_and_packed_grid(tmp_path: Path) -> None:
    image = _save(tmp_path / "image.jpg", Image.new("RGB", (512, 256), (200, 0, 0)), format="JPEG")
    drawn = _save(tmp_path / "mask.png", _mask(512, 256, (256, 0, 512, 256)))
    mask = inpaint.load_mask(str(drawn), str(image), 256, 128, 4)
    assert mask.region.shape == (128, 256)
    assert mask.coverage == pytest.approx(0.5)
    assert mask.latent.shape == (1, (128 // 16) * (256 // 16), 1)
    grid = mask.latent.reshape(128 // 16, 256 // 16)
    assert grid[:, :8].max() == 0.0 and grid[:, 8:].min() == 1.0
    assert mask.original.size == (512, 256)


ENGINE_GLUE = textwrap.dedent(
    """
    import importlib.util, json, sys
    import numpy as np

    spec = importlib.util.spec_from_file_location("image_worker", sys.argv[1])
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    out = {}

    class Scheduler:
        def __init__(self, sigmas):
            self.sigmas = sigmas

    class Holder:
        def __init__(self, sigmas):
            self.scheduler = Scheduler(sigmas)

    rng = np.random.default_rng(3)
    clean, noise, latents = (rng.normal(size=(1, 4, 2)).astype(np.float32) for _ in range(3))
    mask = np.array([1, 0, 1, 0], dtype=np.float32).reshape(1, 4, 1)
    sigmas = np.array([1.0, 0.6, 0.3, 0.0], dtype=np.float32)

    # mflux: the blend is returned as a value, the given array is left alone
    given = latents.copy()
    made = worker._MfluxRepaint(clean, noise, mask)(given, 1, Holder(sigmas))
    expected = mask * latents + (1 - mask) * (0.7 * clean + 0.3 * noise)
    out["mflux_blend"] = bool(np.allclose(made, expected)) and bool(np.array_equal(given, latents))
    last = worker._MfluxRepaint(clean, noise, mask)(latents.copy(), 2, Holder(sigmas))
    out["mflux_last_is_clean"] = bool(np.array_equal(last[0, 1], clean[0, 1]))

    try:
        import torch
    except ImportError:
        out["torch"] = False
    else:
        out["torch"] = True
        t = lambda a: torch.from_numpy(a).to(torch.bfloat16)
        repaint = worker._DiffusersRepaint(t(clean), t(noise), t(mask))
        given = t(latents)
        made = repaint(Holder(torch.from_numpy(sigmas)), 0, given)
        out["diffusers_dtype"] = str(made.dtype)
        out["diffusers_inside"] = bool(torch.equal(made[0, 0], given[0, 0]))
        want = (0.4 * t(clean) + 0.6 * t(noise))[0, 1].float()
        out["diffusers_outside"] = bool(torch.allclose(made[0, 1].float(), want, atol=0.05))
        final = repaint(Holder(torch.from_numpy(sigmas)), 2, given)
        out["diffusers_last_is_clean"] = bool(torch.equal(final[0, 1], t(clean)[0, 1]))
    worker.send("glue", **out)
    """
)


def test_each_engines_step_callback_blends_at_the_sigma_the_latents_reached(tmp_path: Path) -> None:
    script = tmp_path / "glue.py"
    script.write_text(ENGINE_GLUE, encoding="utf-8")
    ran = subprocess.run(
        [sys.executable, str(script), str(WORKER)], capture_output=True, text=True, timeout=300
    )
    assert ran.returncode == 0, ran.stderr
    line = json.loads(ran.stdout.strip().splitlines()[-1])
    assert line["mflux_blend"] and line["mflux_last_is_clean"]
    if line["torch"]:
        assert line["diffusers_dtype"] == "torch.bfloat16"
        assert line["diffusers_inside"] and line["diffusers_outside"] and line["diffusers_last_is_clean"]


MFLUX_LOOP = textwrap.dedent(
    """
    import importlib.util, json, sys
    from types import SimpleNamespace
    import numpy as np

    spec = importlib.util.spec_from_file_location("image_worker", sys.argv[1])
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    out = {}

    class Scheduler:
        def __init__(self, steps):
            self.sigmas = np.linspace(1.0, 0.0, steps + 1).astype(np.float32)
        def scale_model_input(self, latents, t):
            return latents
        def step(self, noise, timestep, latents):
            return latents + noise * (self.sigmas[timestep + 1] - self.sigmas[timestep])

    class Config:
        def __init__(self, steps, start):
            self.scheduler = Scheduler(steps)
            self.time_steps = range(start, steps)
            self.init_time_step = start
            self.guidance = 1.0
            self.width, self.height = 32, 16

    class Ctx:
        def __init__(self):
            self.seen = []
            self.before = self.after = None
        def before_loop(self, latents):
            self.before = latents
        def in_loop(self, t, latents):
            self.seen.append(latents)
        def after_loop(self, latents):
            self.after = latents

    received = []

    def predict(t, latents):
        received.append(latents)
        return np.full_like(latents, 0.5)

    returned = []

    def blend(latents, t, config):
        made = latents * 0 + 100.0 + t
        returned.append(made)
        return made

    ctx = Ctx()
    start = np.zeros((1, 2, 3), dtype=np.float32)
    final = worker._mflux_denoise(start, Config(4, 0), predict, ctx, lambda a: None, blend)
    out["first_step_reads_the_start"] = received[0] is start
    out["each_next_step_reads_the_blend"] = all(received[i + 1] is returned[i] for i in range(3))
    out["callbacks_see_the_blend"] = all(ctx.seen[i] is returned[i] for i in range(4))
    out["final_is_the_last_blend"] = final is returned[-1]

    received.clear()
    plain = worker._mflux_denoise(start, Config(4, 0), predict, Ctx(), lambda a: None)
    out["unmasked_is_plain_euler"] = bool(np.allclose(plain, start - 0.5))

    class Parts:
        mx = SimpleNamespace(eval=lambda *a: None)
        ModelConfig = SimpleNamespace(precision=np.float32)
        class LatentCreator:
            @staticmethod
            def add_noise_by_interpolation(clean, noise, sigma):
                return (1 - sigma) * clean + sigma * noise
        class Qwen21LatentCreator:
            @staticmethod
            def unpack_latents(latents, height, width):
                return ("unpacked", latents)
        class Qwen21PromptEncoder:
            @staticmethod
            def encode_prompt(prompt, prompt_cache, tokenizer, text_encoder):
                return ("embeds", prompt), None
        class VAEUtil:
            @staticmethod
            def decode(vae, latent, tiling_config):
                return latent
        class ImageUtil:
            @staticmethod
            def to_pil(decoded):
                return decoded

    def parts_with(start_step):
        parts = Parts()
        parts.Config = lambda **kw: Config(kw["num_inference_steps"], start_step)
        return parts

    class Transformer:
        def __init__(self):
            self.inputs = []
        def __call__(self, t, config, hidden_states, encoder_hidden_states, encoder_hidden_states_mask):
            self.inputs.append(hidden_states)
            return np.zeros_like(hidden_states)

    def model():
        return SimpleNamespace(
            model_config=None, vae=None, tiling_config=None, prompt_cache={},
            tokenizers={"qwen21": None}, text_encoder=None, transformer=Transformer(),
            callbacks=SimpleNamespace(start=lambda **kw: Ctx()),
        )

    job = SimpleNamespace(width=32, height=16, guidance=1.0, image_path="in.png", image_strength=None,
                          steps=4, seed=1, prompt="p", negative_prompt=None)
    clean = np.full((1, 2, 3), 2.0, dtype=np.float32)
    noise = np.full((1, 2, 3), -1.0, dtype=np.float32)
    mask = np.array([1.0, 0.0], dtype=np.float32).reshape(1, 2, 1)
    repaint = worker._MfluxRepaint(clean, noise, mask)

    m = model()
    picture = worker._mflux_generate(m, job, repaint, parts_with(0))
    out["pure_noise_start"] = bool(np.array_equal(m.transformer.inputs[0], noise))
    out["kept_latent_follows_the_input"] = bool(
        np.allclose(m.transformer.inputs[1][0, 1], 0.25 * clean[0, 1] + 0.75 * noise[0, 1])
    )
    out["decoded_kept_latent_is_clean"] = bool(np.array_equal(picture[1][0, 1], clean[0, 1]))

    m = model()
    job.image_strength = 0.5
    worker._mflux_generate(m, job, repaint, parts_with(2))
    out["strength_start"] = bool(np.allclose(m.transformer.inputs[0], 0.5 * clean + 0.5 * noise))
    out["strength_steps"] = len(m.transformer.inputs)
    worker.send("loop", **out)
    """
)


def test_the_mac_loop_feeds_each_blend_to_the_next_step(tmp_path: Path) -> None:
    script = tmp_path / "loop.py"
    script.write_text(MFLUX_LOOP, encoding="utf-8")
    ran = subprocess.run(
        [sys.executable, str(script), str(WORKER)], capture_output=True, text=True, timeout=300
    )
    assert ran.returncode == 0, ran.stderr
    line = json.loads(ran.stdout.strip().splitlines()[-1])
    for name in (
        "first_step_reads_the_start",
        "each_next_step_reads_the_blend",
        "callbacks_see_the_blend",
        "final_is_the_last_blend",
        "unmasked_is_plain_euler",
        "pure_noise_start",
        "kept_latent_follows_the_input",
        "decoded_kept_latent_is_clean",
        "strength_start",
    ):
        assert line[name] is True, (name, line)
    assert line["strength_steps"] == 2


def test_the_mask_is_handled_the_same_on_the_left_and_the_right(tmp_path: Path) -> None:
    """The outpaint the PC seamed on one side only: 768 wide photo centred on 1024, the mask
    white over both strips and 12 pixels into the photo. Region, feather and latent grid are
    mirror images, so a seam on one side only is not the mask handling."""
    image = _save(tmp_path / "canvas.png", Image.new("RGB", (1024, 768), (90, 120, 60)))
    drawn = Image.new("L", (1024, 768), 0)
    drawn.paste(255, (0, 0, 128 + 12, 768))
    drawn.paste(255, (896 - 12, 0, 1024, 768))
    mask = inpaint.load_mask(str(_save(tmp_path / "border.png", drawn)), str(image), 1024, 768, 8)
    assert np.array_equal(mask.region, mask.region[:, ::-1])
    assert np.allclose(mask.feather, mask.feather[:, ::-1], atol=1e-6)
    grid = mask.latent.reshape(768 // 16, 1024 // 16)
    assert np.array_equal(grid, grid[:, ::-1])
    assert grid[0, :9].min() == 1.0 and grid[0, 9:55].max() == 0.0 and grid[0, 55:].min() == 1.0


PINNED = Path(__file__).resolve().parent / "mflux_pinned_check.py"


def test_the_mac_mask_matches_mflux_0_20_0s_own_latent_code() -> None:
    """Candidates for a Mac-only blend fault, each checked against mflux's own source
    (vendored in fixtures/mflux_0_20_0): token order of the clean latents and the mask on a
    non-square picture with an off-centre box (Qwen21LatentCreator.pack_latents), the noise
    (create_noise), the start (create_for_txt2img_or_img2img, init_time_step), and the sigma
    the kept tokens are at against the sigma the transformer is told (Config, LinearScheduler).
    """
    ran = subprocess.run([sys.executable, str(PINNED)], capture_output=True, text=True, timeout=300)
    assert ran.returncode == 0, ran.stderr
    line = json.loads(ran.stdout.strip().splitlines()[-1])
    for name in (
        "clean_token_is_its_place",
        "mask_token_is_its_place",
        "noise_is_mfluxs_start",
        "clean_unpacks_to_the_encode",
        "noise_start_is_mfluxs",
        "noise_kept_at_the_told_sigma",
        "noise_kept_decodes_to_the_encode",
        "strength_start_is_mfluxs",
        "strength_kept_at_the_told_sigma",
        "strength_kept_decodes_to_the_encode",
    ):
        assert line[name] is True, (name, line)
    assert (line["noise_first_step"], line["noise_steps"]) == (0, 10)
    assert (line["strength_first_step"], line["strength_steps"]) == (4, 6)


def test_the_outside_drift_is_zero_for_the_input_and_large_for_another_picture() -> None:
    rng = np.random.default_rng(4)
    original = Image.fromarray(rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8), mode="RGB")
    region = np.zeros((32, 48), dtype=np.float32)
    region[8:24, 16:32] = 1.0
    changed = np.asarray(original).copy()
    changed[8:24, 16:32] = 0
    assert inpaint.outside_drift(Image.fromarray(changed), original, region) == 0.0
    other = Image.new("RGB", (48, 32), (0, 0, 0))
    assert inpaint.outside_drift(other, original, region) > 50
