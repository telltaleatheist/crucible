"""Runs the Mac arm's mask code against mflux 0.20.0's own latent code, in a subprocess.

mflux needs MLX, which is macOS-only, so `mlx.core` is stood in for by numpy (float32 for
bf16). The mflux files are vendored verbatim from the 0.20.0 wheel under
fixtures/mflux_0_20_0 (MIT, see LICENSE.txt there): the config (init_time_step, time_steps),
the linear scheduler (the shifted sigmas and the Euler step), Qwen21LatentCreator (noise,
pack, unpack) and LatentCreator (encode_image, the image-to-image start). Only the VAE and
the picture loading are stubbed: the stub VAE returns latents that encode their own position,
so a token that lands anywhere but its place shows.

Prints one JSON line of named checks. Run by tests/test_image_inpaint.py.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VENDORED = HERE / "fixtures" / "mflux_0_20_0"
WORKER = HERE.parent / "crucible" / "jobs" / "image" / "worker.py"


def _mlx_on_numpy() -> types.ModuleType:
    core = types.ModuleType("mlx.core")
    core.float32 = np.float32
    core.bfloat16 = np.float32
    core.Dtype = object
    core.reshape = np.reshape
    core.transpose = np.transpose
    core.concatenate = lambda arrays, axis=0: np.concatenate(arrays, axis=axis)
    core.zeros = lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
    core.linspace = lambda a, b, n: np.linspace(a, b, n, dtype=np.float32)
    core.exp = np.exp
    core.broadcast_to = np.broadcast_to
    core.arange = lambda n, dtype=np.float32: np.arange(n, dtype=dtype)
    core.array = lambda a, dtype=None: np.array(a, dtype=dtype)
    core.eval = lambda *arrays: None
    core.random = types.SimpleNamespace(
        key=lambda seed: seed,
        normal=lambda shape, key: np.random.default_rng(key).standard_normal(shape).astype(np.float32),
    )
    mlx = types.ModuleType("mlx")
    mlx.core = core
    mlx.nn = types.ModuleType("mlx.nn")
    sys.modules.update({"mlx": mlx, "mlx.core": core, "mlx.nn": mlx.nn})
    return core


def _stub(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _vendored(name: str, file: str) -> types.ModuleType:
    loader = importlib.machinery.SourceFileLoader(name, str(VENDORED / file))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


class Qwen21ModelConfig:
    """ModelConfig.qwen_image_21() in mflux 0.20.0's model_config.py."""

    precision = np.float32
    requires_sigma_shift = True
    sigma_base_shift = 0.5
    sigma_max_shift = 0.9
    sigma_base_seq_len = 256
    sigma_max_seq_len = 8192
    sigma_shift_terminal = 0.02


HEIGHT, WIDTH, CHANNELS = 96, 160, 64
LATENT_H, LATENT_W = HEIGHT // 16, WIDTH // 16


def _encoded() -> np.ndarray:
    """(1, 64, h, w) like Qwen21VAE.encode, each value naming its channel, row and column."""
    c, y, x = np.meshgrid(np.arange(CHANNELS), np.arange(LATENT_H), np.arange(LATENT_W), indexing="ij")
    return (1000.0 * y + 10.0 * x + c / 100.0).astype(np.float32)[None]


def main() -> None:
    core = _mlx_on_numpy()
    for name in (
        "mflux", "mflux.models", "mflux.models.common", "mflux.models.qwen21",
        "mflux.models.qwen21.latent_creator", "mflux.models.common.latent_creator",
        "mflux.models.common.vae", "mflux.utils",
    ):
        _stub(name)
    _stub("mflux.models.common.config", ModelConfig=Qwen21ModelConfig)
    _stub("mflux.models.common.config.model_config", ModelConfig=Qwen21ModelConfig)
    _stub("mflux.models.common.schedulers", SCHEDULER_REGISTRY={}, try_import_external_scheduler=None)
    _stub("mflux.utils.dimension_resolver", DimensionResolver=None)
    _stub("mflux.utils.scale_factor", ScaleFactor=None)
    encoded = _encoded()
    _stub(
        "mflux.models.common.vae.vae_util",
        VAEUtil=types.SimpleNamespace(
            encode=lambda vae, image, tiling_config=None: encoded,
            decode=lambda vae, latent, tiling_config=None: latent,
        ),
    )
    _stub(
        "mflux.utils.image_util",
        ImageUtil=types.SimpleNamespace(
            load_image=lambda path: types.SimpleNamespace(convert=lambda mode: "picture"),
            scale_to_dimensions=lambda image, target_width, target_height: image,
            to_array=lambda image: None,
            to_pil=lambda decoded: decoded,
        ),
    )
    _vendored("mflux.models.common.schedulers.base_scheduler", "base_scheduler.py.txt")
    _vendored("mflux.models.common.schedulers.linear_scheduler", "linear_scheduler.py.txt")
    q21 = _vendored(
        "mflux.models.qwen21.latent_creator.qwen21_latent_creator", "qwen21_latent_creator.py.txt"
    ).Qwen21LatentCreator
    creator = _vendored("mflux.models.common.latent_creator.latent_creator", "latent_creator.py.txt")
    config_module = _vendored("mflux.models.common.config.config", "config.py.txt")

    spec = importlib.util.spec_from_file_location("image_worker", WORKER)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    inpaint = worker.inpaint

    region = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    region[20:50, 36:90] = 1.0
    grid = inpaint.latent_grid(region)
    mask = inpaint.Mask(region=region, feather=region, latent=inpaint.packed(grid), original=None, grid=grid)
    out: dict = {}

    engine = object.__new__(worker.MfluxEngine)
    engine._mx = core
    model = types.SimpleNamespace(vae=None, tiling_config=None, model_config=Qwen21ModelConfig())

    def job(strength):
        return types.SimpleNamespace(
            image_path="in.png", height=HEIGHT, width=WIDTH, seed=7, steps=10, guidance=1.0,
            image_strength=strength, prompt="p", negative_prompt=None, mask=mask,
        )

    repaint = engine._repaint(model, job(None))
    tokens = np.arange(LATENT_H * LATENT_W)
    rows, cols = tokens // LATENT_W, tokens % LATENT_W
    out["clean_token_is_its_place"] = bool(
        np.array_equal(repaint.clean[0], encoded[0][:, rows, cols].T)
    )
    out["mask_token_is_its_place"] = bool(np.array_equal(repaint.mask[0, :, 0], grid[rows, cols]))
    out["noise_is_mfluxs_start"] = bool(np.array_equal(repaint.noise, q21.create_noise(7, HEIGHT, WIDTH)))
    unpacked = q21.unpack_latents(repaint.clean, HEIGHT, WIDTH)[:, :, 0]
    out["clean_unpacks_to_the_encode"] = bool(np.array_equal(unpacked, encoded))

    parts = types.SimpleNamespace(
        mx=core,
        ModelConfig=Qwen21ModelConfig,
        Config=config_module.Config,
        LatentCreator=creator.LatentCreator,
        VAEUtil=sys.modules["mflux.models.common.vae.vae_util"].VAEUtil,
        Qwen21LatentCreator=q21,
        Qwen21PromptEncoder=types.SimpleNamespace(encode_prompt=lambda **kw: ("embeds", None)),
        ImageUtil=sys.modules["mflux.utils.image_util"].ImageUtil,
    )
    kept = grid[rows, cols] == 0

    for strength, label in ((None, "noise"), (0.4, "strength")):
        seen = []

        def transformer(t, config, hidden_states, encoder_hidden_states, encoder_hidden_states_mask):
            seen.append((t, float(config.scheduler.sigmas[t]), hidden_states.copy()))
            return np.zeros_like(hidden_states)

        class Ctx:
            def before_loop(self, latents):
                pass

            def in_loop(self, t, latents):
                pass

            def after_loop(self, latents):
                pass

        model.transformer = transformer
        model.prompt_cache, model.tokenizers, model.text_encoder = {}, {"qwen21": None}, None
        model.callbacks = types.SimpleNamespace(start=lambda **kw: Ctx())
        the_job = job(strength)
        picture = worker._mflux_generate(model, the_job, engine._repaint(model, the_job), parts)
        config = config_module.Config(
            model_config=Qwen21ModelConfig(), num_inference_steps=10, height=HEIGHT, width=WIDTH,
            guidance=1.0, image_path="in.png" if strength else None, image_strength=strength,
        )
        mflux_start = creator.LatentCreator.create_for_txt2img_or_img2img(
            seed=7, height=HEIGHT, width=WIDTH,
            img2img=creator.Img2Img(
                vae=None, latent_creator=q21, sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step, image_path=config.image_path,
            ),
        )
        out[f"{label}_start_is_mfluxs"] = bool(np.allclose(seen[0][2], mflux_start))
        out[f"{label}_first_step"] = seen[0][0]
        levels = [
            np.allclose(hidden[0, kept], (1 - sigma) * repaint.clean[0, kept] + sigma * repaint.noise[0, kept], atol=1e-5)
            for _, sigma, hidden in seen
        ]
        out[f"{label}_kept_at_the_told_sigma"] = bool(all(levels))
        out[f"{label}_steps"] = len(seen)
        final = picture[:, :, 0]
        kept_pixels = grid == 0
        out[f"{label}_kept_decodes_to_the_encode"] = bool(
            np.allclose(final[0][:, kept_pixels], encoded[0][:, kept_pixels], atol=1e-3)
        )
    worker.send("pinned", **out)


if __name__ == "__main__":
    main()
