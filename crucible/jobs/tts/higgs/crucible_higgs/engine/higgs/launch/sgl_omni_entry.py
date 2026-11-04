#!/usr/bin/env python3
"""narrator's entry point for the SGLang-Omni server: set the context, then
hand off to sgl-omni's own CLI, unchanged.

WHY THIS FILE EXISTS. SGLang-Omni's Higgs builder hard-codes its context window
as a CLASS ATTRIBUTE:

    sglang_omni/models/higgs_tts/engine_builder.py:29
        class HiggsTtsEngineBuilder:
            context_length = 4096

There is no flag and no config path for it. Three attempts are recorded
(2026-09-09, the ladder's author, sglang_omni 0.1.4 / sglang 0.5.18 - the same
versions Crucible's tts env carries):

  * `--tts_engine.factory.context_length` dies with "factory does not accept a
    'context_length' parameter" - the field exists on FactoryArgs but the
    factory function does not take it.
  * `--tts_engine.engine.context_length` collides: it is already passed as an
    explicit keyword.
  * There is no thinker stage to override it on.

The only thing that has ever worked is changing the attribute - which they did
by REWRITING THAT LINE in site-packages with a sed script, and which the
2026-09-15 env rebuild silently wiped, taking every 8192-token render with it.
narrator does not patch site-packages (this stack's whole selling point is that
it needs no patches - `sgl_served.py`'s header says so). It does the one other
thing that reaches a class attribute: ASSIGNS IT, in the server process, after
the module is imported and before the engine is constructed. That is what this
file is. `sgl-omni`'s own `app()` runs directly afterwards and is not wrapped,
patched or re-implemented.

STANDALONE ON PURPOSE. This runs under the SGLang env's interpreter
(`$HIGGS_SGL_ENV/bin/python`), which is a different environment from the one
narrator is installed in and has never heard of `narrator`. So: stdlib only, no
package-relative imports, and the env var is parsed here rather than handed
down.

IT SAYS WHAT IT DID, on stderr, which is the server log narrator already reads
and the doctor already quotes. A context length is the difference between a
chunk that renders and an HTTP 500 from inside the scheduler; "which window is
this server actually serving" must be answerable from the log rather than from
whether someone remembered to set a variable.
"""
import os
import sys

#: The same name every Higgs backend takes, and the one Crucible sets per voice
#: in the spawn environment (`narrator/engine/higgs/served_common.py`:
#: `CONTEXT_LENGTH_ENV`). Duplicated as a literal rather than imported for the
#: reason in the module docstring: this process cannot import narrator.
CONTEXT_LENGTH_ENV = 'HIGGS_CONTEXT_LENGTH'

#: `engine_builder.py:29`. Unset means this, which is what the builder would
#: have used anyway - so an unset variable changes not one byte of the launch.
BUILDER_DEFAULT_CONTEXT_LENGTH = 4096


def wanted_context_length() -> int:
    """The context this launch asks for. Refused, never coerced - the same rule
    `served_common.context_length` keeps on narrator's side of the wire."""
    raw = (os.environ.get(CONTEXT_LENGTH_ENV) or '').strip()
    if not raw:
        return BUILDER_DEFAULT_CONTEXT_LENGTH
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(
            f'{CONTEXT_LENGTH_ENV}={raw!r} is not a whole number of tokens.')
    if value < 1:
        raise SystemExit(f'{CONTEXT_LENGTH_ENV}={raw!r} must be at least 1.')
    return value


def apply_context_length(value: int) -> int:
    """Assign `HiggsTtsEngineBuilder.context_length`, and report what was there.

    A MISSING BUILDER IS A HARD ERROR, not a shrug: reaching this file at all
    means the caller asked narrator to launch the SGLang-Omni Higgs stack, and
    an import error here means the env is not that stack. Starting the server
    anyway would serve 4096 under a request for something else, which is the
    exact silence this file exists to end.
    """
    try:
        from sglang_omni.models.higgs_tts.engine_builder import HiggsTtsEngineBuilder
    except Exception as exc:                       # noqa: BLE001 - reported whole
        raise SystemExit(
            'narrator could not import '
            'sglang_omni.models.higgs_tts.engine_builder.HiggsTtsEngineBuilder '
            f'({type(exc).__name__}: {exc}). This entry point exists to set that '
            "class's `context_length` before the engine is built; without it the "
            'server would silently serve the built-in 4096-token window.')
    before = getattr(HiggsTtsEngineBuilder, 'context_length', None)
    HiggsTtsEngineBuilder.context_length = int(value)
    return before


def main(argv=None) -> int:
    value = wanted_context_length()
    before = apply_context_length(value)
    print(f'[narrator] HiggsTtsEngineBuilder.context_length {before} -> {value} '
          f'(from {CONTEXT_LENGTH_ENV}; upstream engine_builder.py:29 has no flag)',
          file=sys.stderr, flush=True)
    from sglang_omni.cli import app
    if argv is not None:
        sys.argv = [sys.argv[0]] + list(argv)
    return app()


if __name__ == '__main__':
    sys.exit(main())
