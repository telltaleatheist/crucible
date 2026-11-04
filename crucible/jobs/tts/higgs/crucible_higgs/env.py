"""ONE reader for every number the worker takes out of the environment.

Moved from narrator@dc05e7e7 `env.py`, body unchanged.

WHY IT IS HERE AND NOT IN A PACKAGE. It was written in
`engine/higgs/mlx_backend.py` (`_env_number`, 2026-09-08) and it states a
POLICY, not an MLX fact: garbage is refused by name, never coerced and never
defaulted past. A policy with one implementation inside one backend is a policy
the rest of the tree cannot obey, and `serve/worker.py` did not - four of its
reads sat behind `except ValueError: <the default>`, so
`ORPHEUS_STREAM_BATCH='1 6'` (a stray space) rendered every Listen batch 16 wide
on a box tuned for 4 and said nothing. One fact, two owners, nothing comparing
them - ARCHITECTURE.md's R1, inside a single package.

This module is the owner. It is TOP-LEVEL and imports `os` and nothing else, so
every layer can reach it without a cycle: `narrator.engine.*` cannot import
`narrator.serve`, `narrator.serve` deliberately keeps `narrator.engine` out of
its module scope (the engine import is deferred until after `ready` is sent),
and `narrator/__init__.py` costs nothing. A home inside either package would
have been a home the other one could not use.

    from ..env import env_number
    width = env_number('NARRATOR_HIGGS3_MLX_BATCH', 1, int, 1,
                       'the MLX batch ceiling, in rows')

WHAT IS AND IS NOT A DEFECT. Having a default is fine - a shipped knob that
nobody set has to mean something, and the number narrator ships with is a
decision this package is allowed to make. COERCING GARBAGE TO IT is the defect,
because it turns an operator's typo into a silent re-tuning of the run. The two
are distinguished here: unset (or exported empty, which is what a shell hands
over when a setting was cleared) takes the default; anything that is not a
number, or is out of range, RAISES and names the variable.
"""
import os

__all__ = ['env_number']


def env_number(name: str, default, cast, minimum, what: str):
    """One env variable -> a number, or a ValueError NAMING THE VARIABLE.

    Garbage is never coerced and never defaulted past: a width or a budget read
    from a typo would silently render a whole book at the wrong memory profile,
    and "it fell back to the default" is exactly the sentence nobody can debug.

    `name` is the variable; `default` the value an UNSET or empty one takes;
    `cast` the one-argument converter (`int` refuses '1.5', which is the point);
    `minimum` the smallest value that means anything (a width of 0 is not a
    narrower batch, it is a typo); `what` a phrase completing "It is ..." so the
    refusal says what the operator was configuring.
    """
    raw = (os.environ.get(name) or '').strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f'{name}={raw!r} is not a number. It is {what}; unset it for the '
            f'default ({default:g}).') from None
    if value < minimum:
        raise ValueError(
            f'{name}={raw!r} is below {minimum:g}. It is {what}; unset it for '
            f'the default ({default:g}).')
    return value
