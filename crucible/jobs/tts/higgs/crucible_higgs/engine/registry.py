"""Engine id -> the class that implements it and the config that builds it.

ONE id: `higgs-v3`, Higgs Audio v3 (`crucible_higgs.engine.higgs`). Moved from
narrator@dc05e7e7 `engine/registry.py`, where it sat beside `orpheus` (retired,
Owen 2026-09-14) and `higgs-v2-scaffold` (never shipped); neither came along.

The id is what `NARRATOR_ENGINE` carries into `python -m crucible_higgs.serve`
(`crucible/engines/narrator.py` sets it on every spawn).

WHY FACTORIES AND NOT A DICT OF CLASSES. Importing an engine class imports its
backend module, and a backend module is where the heavy imports live. The
registry must be readable - "which engines exist?", "is 'higgs-v3' known?" - on
an interpreter with neither torch nor transformers installed. So the values are
zero-argument callables that import on demand, and `ids()` never imports
anything.

NO FALLBACK. An unknown id RAISES, naming the id and listing the known ones: an
engine substitution is a whole book rendered by the wrong model and reported as
success.
"""


#: `higgs-v3` is ONE engine with TWO backends, chosen by PLATFORM - the id, the
#: voice document, the geometry, the budget and the stop policy are identical
#: and only the runtime differs:
#:
#:   darwin       IN-PROCESS through mlx-audio's `higgs_audio_v3`. Apple Silicon
#:                runs the whole model natively, so the weights load into the
#:                worker and there is no server between a chunk and its audio.
#:                Owen, 2026-09-05: the Mac's Listen streaming is used
#:                constantly and needs Higgs.
#:   everything   SERVED by vllm-omni over HTTP. On Windows/Linux there is no
#:   else         choice: v3 has no HF modeling class, and vllm-omni's torch
#:                cannot share an env with Orpheus's vLLM 0.7.3.
#:
#: This is a function of `sys.platform` alone and takes no arguments beyond it,
#: so a test can ask for either arm on any machine.
HIGGS_V3_BACKENDS = ('inprocess-mlx', 'served')


def higgs_v3_backend_for_platform(platform_name: str) -> str:
    """'inprocess-mlx' on darwin, 'served' anywhere else.

    NOT a capability probe: whether mlx-audio happens to be importable is a
    question about an ENVIRONMENT, and answering it here would let a Mac with a
    broken install silently fall through to a served backend whose server does
    not exist on that machine. The platform decides; the backend then fails
    loudly if its runtime is missing (`detect_backend` imports mlx_audio).
    """
    name = (platform_name or '').strip().lower()
    if not name:
        raise ValueError(
            'higgs_v3_backend_for_platform() needs a platform name (sys.platform); '
            'an empty one cannot choose between an in-process model and a server.')
    return 'inprocess-mlx' if name.startswith('darwin') else 'served'


def _higgs_v3_engine():
    import sys
    if higgs_v3_backend_for_platform(sys.platform) == 'inprocess-mlx':
        from .higgs import HiggsV3MlxEngine
        return HiggsV3MlxEngine
    from .higgs import HiggsV3Engine
    return HiggsV3Engine


def _higgs_v3_config(**kwargs):
    import sys
    if higgs_v3_backend_for_platform(sys.platform) == 'inprocess-mlx':
        from .higgs import higgs_v3_mlx_config_from_worker_kwargs
        return higgs_v3_mlx_config_from_worker_kwargs(**kwargs)
    from .higgs import higgs_v3_config_from_worker_kwargs
    return higgs_v3_config_from_worker_kwargs(**kwargs)


# id -> (engine class factory, config factory). Both are called with no imports
# performed until then; see the module docstring.
ENGINES = {
    'higgs-v3': (_higgs_v3_engine, _higgs_v3_config),
}


def ids():
    """Every known engine id, sorted. Imports nothing."""
    return sorted(ENGINES)


def _entry(engine_id: str):
    if engine_id not in ENGINES:
        raise ValueError(
            f"Unknown narrator engine '{engine_id}'. Known engines: "
            f"{', '.join(ids())}. Refusing to substitute one - rendering a book "
            'with a different model than the one asked for is a silent failure.')
    return ENGINES[engine_id]


def engine_class(engine_id: str):
    """The Engine implementation for `engine_id`. Imports its backend module."""
    return _entry(engine_id)[0]()


def engine_config(engine_id: str, **kwargs):
    """The config object `engine_class(engine_id)` takes, built from the
    worker's keyword arguments. Each engine's factory decides which keywords it
    understands and REFUSES the rest by name."""
    return _entry(engine_id)[1](**kwargs)
