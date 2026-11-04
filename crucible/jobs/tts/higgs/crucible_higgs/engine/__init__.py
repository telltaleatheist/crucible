"""crucible_higgs.engine - the engine seam and the Higgs engine behind it.

    engine/
      protocol.py      Engine, Codec, VoiceRef, Budget, StopPolicy, BackendSpec,
                       ServedBackend  (typing.Protocol + dataclasses; no torch)
      registry.py      engine id -> (engine class, config) factories; one id,
                       'higgs-v3'
      item_sampling.py the per-item rung parser (`sampling` and `take`)
      log.py           the one log stream (stderr; stdout is the wire)
      higgs/           Higgs v3, served (SGLang-Omni / vllm-omni) or in-process
                       on MLX

Moved from narrator@dc05e7e7 `engine/`. narrator's Orpheus engine and the
re-exports and import-path aliases it carried here are gone: this package
serves Higgs v3 and nothing else.

Importing this package still costs no torch, no vLLM, no mlx and no
transformers: every backend defers its heavy imports to the function that needs
them, and the registry's values are factories that import on demand.

    from crucible_higgs.engine import registry
    engine = registry.engine_class('higgs-v3')(registry.engine_config('higgs-v3', ...))
"""
from . import protocol, registry
from .protocol import (
                       BackendSpec,
                       Budget,
                       ClipsVoice,
                       Codec,
                       DescriptionVoice,
                       Engine,
                       ReferenceClip,
                       ServedBackend,
                       SpeechRequest,
                       StopPolicy,
                       TokenVoice,
)

__all__ = [
    'BackendSpec',
    'Budget',
    'ClipsVoice',
    'Codec',
    'DescriptionVoice',
    'Engine',
    'ReferenceClip',
    'ServedBackend',
    'SpeechRequest',
    'StopPolicy',
    'TokenVoice',
    'protocol',
    'registry',
]
