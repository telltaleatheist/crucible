"""YuE2's own score tools, vendored unchanged from the yue2-music instrumental skill.

Source: https://github.com/multimodal-art-projection/YuE at 18a07bb628f070c2eede44c3143834818e856a73
(the commit Crucible's yue2-infer pin installs), skills/yue2-music/instrumental/scripts/:
abc_tools.py, compile_score.py, common.py, instrumentalize.py, instrumental.py. MIT licence (LICENSE here).

They import one another by bare module name, so yue2_worker puts this directory on sys.path.
Re-vendor them with the yue2-infer pin; do not edit them in place.
"""
