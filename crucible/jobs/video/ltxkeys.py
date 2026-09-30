"""From LTX-2.5's original tensor names (what a ComfyUI GGUF carries) to diffusers'.

diffusers' own single-file converter (`convert_ltx2_transformer_to_diffusers`
at the pinned commit) has only the LTX-2.0 table, so it leaves 2.3/2.5's
`prompt_adaln_single` and `audio_prompt_adaln_single` unrenamed and the load
would miss them. This is the table the pinned commit's
scripts/convert_ltx2_to_diffusers.py uses for version 2.5
(`LTX_2_3_TRANSFORMER_KEYS_RENAME_DICT` plus
`LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP`, loaded there with strict=True),
restated here so the worker can hand diffusers a state dict whose names
already match and whose connector tensors are gone (the connectors come from
the Lightricks repo's own connectors/ folder).

Pure Python on purpose: the tests check it against the real names in the
pinned GGUF's header without torch.
"""

from __future__ import annotations

RENAMES: tuple[tuple[str, str], ...] = (
    ("patchify_proj", "proj_in"),
    ("audio_patchify_proj", "audio_proj_in"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
)

CONNECTOR_PREFIXES: tuple[str, ...] = (
    "video_embeddings_connector",
    "audio_embeddings_connector",
    "transformer_1d_blocks",
    "text_embedding_projection",
    "connectors.",
    "video_connector",
    "audio_connector",
    "text_proj_in",
)

CHECKPOINT_PREFIX = "model.diffusion_model."

TIME_EMBEDS: tuple[tuple[str, str], ...] = (
    ("adaln_single.", "time_embed."),
    ("audio_adaln_single.", "audio_time_embed."),
)


def diffusers_name(name: str) -> str | None:
    """The diffusers parameter a transformer tensor loads into; None for a connector's."""
    if name.startswith(CHECKPOINT_PREFIX):
        name = name[len(CHECKPOINT_PREFIX):]
    if name.startswith(CONNECTOR_PREFIXES):
        return None
    for old, new in RENAMES:
        name = name.replace(old, new)
    if ".weight" in name or ".bias" in name:
        for old, new in TIME_EMBEDS:
            if name.startswith(old):
                return new + name[len(old):]
    return name


__all__ = ["CONNECTOR_PREFIXES", "RENAMES", "diffusers_name"]
