"""crucible_higgs - Crucible's Higgs v3 narration worker and the engine it drives.

    python -m crucible_higgs.serve

run by `crucible/engines/narrator.py` in the tts env's own interpreter, with
`PYTHONPATH` naming the directory this package sits in. It is a TOP-LEVEL package
there, not `crucible.*`: the tts env has torch, sglang and mlx-audio and does
not have Crucible's server dependencies, so nothing here imports `crucible`, and
nothing in Crucible's server imports this.

Moved from BookForge's `narrator` package at dc05e7e7 (the sha the tts recipes
pinned), Higgs only: `serve/` (the NDJSON worker), `engine/` (the protocol and
Higgs v3, served and MLX), `env.py` (the one env-number reader) and three pieces
of `text/` (the gap rule, the sentence splitter the split ladder cuts on, the
unspoken-tag pattern). Importing this package costs nothing.
"""
