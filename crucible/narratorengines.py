from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .backend import CUDA_LINUX, MLX_DARWIN

HIGGS_V3 = "higgs-v3"

VOICES_PULL_COMMAND = "crucible voices pull"

NARRATOR_ENGINE_SAMPLING: dict[str, dict[str, float]] = {
    HIGGS_V3: {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
}

NARRATOR_ENGINES: frozenset[str] = frozenset(NARRATOR_ENGINE_SAMPLING)

DOCUMENT_READERS: frozenset[str] = frozenset({HIGGS_V3})

ESTIMATE_BASES: frozenset[str] = frozenset({"measured", "declared"})


class VoicesDocumentView(Protocol):

    path: Path

    def environment(self) -> dict[str, str]:
        ...

    def weights_for(self, voice: str) -> Path:
        ...


@dataclass(frozen=True)
class EngineFootprint:

    engine: str
    memory_bytes_estimate: int
    estimate_basis: str
    estimate_note: str | None
    max_num_seqs: int
    max_num_seqs_note: str
    mem_fraction: float | None = None
    mem_fraction_note: str | None = None
    context_length: int | None = None
    context_length_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "estimate_basis": self.estimate_basis,
        }
        if self.estimate_note is not None:
            document["estimate_note"] = self.estimate_note
        document["max_num_seqs"] = self.max_num_seqs
        document["max_num_seqs_note"] = self.max_num_seqs_note
        for key in (
            "mem_fraction", "mem_fraction_note",
            "context_length", "context_length_note",
        ):
            value = getattr(self, key)
            if value is not None:
                document[key] = value
        return document


def declared_tts_footprints(backend_kind: str) -> tuple[EngineFootprint, ...]:
    if backend_kind == CUDA_LINUX:
        return (
            EngineFootprint(
                engine=HIGGS_V3,
                memory_bytes_estimate=19_000_000_000,
                estimate_basis="declared",
                estimate_note=(
                    "SGLang-Omni's configured reservation on this arm, not a "
                    "watched card: `--mem-fraction-static 0.60 holds ~19 GB at 16 "
                    "in flight` (BookForge higgs-models.json "
                    "serving.sglang._memFractionStaticNote, owens-pc RTX 3090 Ti, "
                    "2026-09-05). It is the SERVER's footprint rather than any "
                    "one checkpoint's, which is why every Higgs v3 voice on this "
                    "arm declared it until 2026-09-19 and why it is stated once "
                    "here now. Owed: a real reading on this card."
                ),
                max_num_seqs=16,
                max_num_seqs_note=(
                    "16 is vllm-omni's OWN stage-0 value in "
                    "higgs_multimodal_qwen3.yaml, and a measured ceiling at the "
                    "shipped memory fractions: on owens-pc (RTX 3090 Ti, "
                    "2026-09-05) 16 concurrent at 0.35 + 0.10 ran 11,387-11,584 "
                    "chars/min over three runs while 32 filled the card and "
                    "stalled. THE deathstalker CAP CERTIFICATE RAN AT 64, at the "
                    "older fractions and before that stall was measured, so 16 is "
                    "not the width its cap was certified at; nothing measured says "
                    "whether batch width moves the safe chunk length, and if it "
                    "does, that certificate is bound to 64 and this is the field "
                    "that would have to change. It is also the width of narrator's "
                    "own batch (v3_served.serve_concurrency), so raising it raises "
                    "concurrent POSTs and VRAM pressure together."
                ),
            ),
        )
    if backend_kind == MLX_DARWIN:
        return (
            EngineFootprint(
                engine=HIGGS_V3,
                memory_bytes_estimate=12_133_000_000,
                estimate_basis="declared",
                estimate_note=(
                    "The one recorded MLX figure for Higgs v3 weights of this "
                    "shape — 11.3 GiB peak at a 900-character chunk, from "
                    "deathstalker's MLX cap certificate (mlx-audio 0.4.8 / mlx "
                    "0.32.0 on owens-mac-studio, 2026-09-05). Somebody else's "
                    "reading carried across, which is exactly what 'declared' "
                    "means. Owed: watch this machine's own allocator."
                ),
                max_num_seqs=16,
                max_num_seqs_note=(
                    "16 is vllm-omni's own stage-0 value and the width narrator "
                    "batches at (v3_served.serve_concurrency), which is the number "
                    "every packaged voice declared until 2026-09-19. The mlx arm "
                    "starts no server under narrator, so what this sizes here is "
                    "narrator's own batch. Owed: a width sweep on this machine."
                ),
            ),
        )
    return ()
