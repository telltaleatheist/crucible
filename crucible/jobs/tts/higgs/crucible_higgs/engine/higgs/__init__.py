"""crucible_higgs.engine.higgs - Higgs Audio v3, the one narration engine.

Moved from narrator@dc05e7e7 `engine/higgs/` (Crucible owns the TTS model code
now, the way it owns its YuE2 and LTX workers). `higgs-v3` is ONE engine with
TWO arms, chosen by platform in `engine/registry.py`:

  served      `v3_engine.HiggsV3Engine` over `sgl_served` (SGLang-Omni, the stack
              the shipped voices select) or `v3_served` (vllm-omni) - a
              `BackendSpec.kind == 'served'` backend the worker launches,
              health-checks and stops, out of `launch/`.
  in-process  `mlx_backend.HiggsV3MlxEngine` on Apple Silicon, through mlx-audio.

`codec.py` and `config.py` carry the geometry, the voice document reader
(`load_voice`) and `FrameMeasure`, which both arms share. narrator's Higgs v2
scaffold (`engine.py`, `transformers_backend.py`, registry id
`higgs-v2-scaffold`, never shipped) did not come along, and neither did
anything Orpheus.

Licence, v3: Boson Higgs TTS 3 Research and Non-Commercial - fine for personal
use and, under the Creator Use Grant, for credited creator content; production
deployment or embedding in a product needs separate licensing.

Importing this package costs no torch and no transformers.
"""
from .codec import HiggsCodec, HiggsStreamMisaligned
from .config import (
                     HiggsBudget,
                     HiggsConfig,
                     HiggsDefaults,
                     higgs_config_from_worker_kwargs,
                     higgs_stop_policy,
                     load_voice,
                     load_voices,
)
from .mlx_backend import (
                     FrameFilterReport,
                     HiggsMlxStreamMisaligned,
                     HiggsV3MlxBudget,
                     HiggsV3MlxCodec,
                     HiggsV3MlxConfig,
                     HiggsV3MlxEngine,
                     higgs_v3_mlx_config_from_worker_kwargs,
                     higgs_v3_mlx_stop_policy,
                     real_code_frames,
                     revert_delay_pattern,
)
from .prompt import build_conversation
from .served_common import serving_stack
from .sgl_served import HiggsSglServedBackend
from .v3_engine import (
                     HiggsV3Budget,
                     HiggsV3Codec,
                     HiggsV3Config,
                     HiggsV3Defaults,
                     HiggsV3Engine,
                     higgs_v3_config_from_worker_kwargs,
                     higgs_v3_stop_policy,
)
from .v3_served import HiggsV3ServedBackend, HiggsV3ServerError

__all__ = [
    'HiggsBudget',
    'HiggsCodec',
    'HiggsConfig',
    'HiggsDefaults',
    'HiggsStreamMisaligned',
    'HiggsV3Budget',
    'HiggsV3Codec',
    'HiggsV3Config',
    'HiggsV3Defaults',
    'HiggsV3Engine',
    'HiggsV3MlxBudget',
    'HiggsV3MlxCodec',
    'HiggsV3MlxConfig',
    'HiggsV3MlxEngine',
    'HiggsMlxStreamMisaligned',
    'FrameFilterReport',
    'HiggsSglServedBackend',
    'HiggsV3ServedBackend',
    'HiggsV3ServerError',
    'build_conversation',
    'serving_stack',
    'higgs_v3_config_from_worker_kwargs',
    'higgs_v3_mlx_config_from_worker_kwargs',
    'higgs_v3_mlx_stop_policy',
    'real_code_frames',
    'revert_delay_pattern',
    'higgs_v3_stop_policy',
    'higgs_config_from_worker_kwargs',
    'higgs_stop_policy',
    'load_voice',
    'load_voices',
]
