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

# How each engine's chunks join, which is the model's own behaviour and what an assembler reads
# (GET /v1/voices `edges`). `pads` False: it emits bare speech, and the assembler fades each
# chunk's edges and inserts the gaps. A Higgs edge sits near -30 dB after the codec's trim and
# clicks on a join; 10 in / 25 out takes it to -45..-48 dB, and the tail needs the wider window.
# tests/test_voice_edges.py holds these equal to the worker's own constants, which it cannot
# import from here: it runs in the tts env.
NARRATOR_ENGINE_EDGES: dict[str, dict[str, bool | float]] = {
    HIGGS_V3: {"pads": False, "fade_in_ms": 10.0, "fade_out_ms": 25.0},
}

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
                    "SGLang-Omni's configured reservation on this arm as first "
                    "stated: `--mem-fraction-static 0.60 holds ~19 GB at 16 in "
                    "flight` (BookForge higgs-models.json "
                    "serving.sglang._memFractionStaticNote, owens-pc RTX 3090 Ti, "
                    "2026-09-05). KNOWN LOW since narrator's fp32 vocoder (2.0.24): "
                    "on owens-pc 2026-11-02, 16 in flight held 22,290 MiB "
                    "(23.4e9 bytes) in the WSL VM. Not raised to that reading "
                    "because it is a 16-wide figure, the width is now 12, and a "
                    "23.4e9 estimate would refuse every load while the desktop "
                    "holds more than ~2.4 GB. It is the SERVER's footprint rather "
                    "than any one checkpoint's, so it is stated once here. Owed: "
                    "a reading at 12 on this card."
                ),
                max_num_seqs=12,
                max_num_seqs_note=(
                    "12 since 2026-11-02, down from 16, chosen to FIT and not yet "
                    "speed-measured. On owens-pc (RTX 3090 Ti 24 GB) 2026-11-02 a "
                    "fine-tune at 16 in flight with mem-fraction-static 0.60 and "
                    "the fp32 vocoder held 22,290 MiB in the WSL VM, the card "
                    "reached 24,024 of 24,564 MiB with the desktop at ~1.7 GiB, "
                    "and ~0.45 GiB spilled to shared memory - the WDDM paging "
                    "2026-09-19 measured as 4-10x slower with no error. The 16 "
                    "before it was vllm-omni's own stage-0 value and a measured "
                    "ceiling at the OLD fractions (11,387-11,584 chars/min, "
                    "2026-09-05), taken before the fp32 vocoder. THE deathstalker "
                    "CAP CERTIFICATE RAN AT 64; nothing measured says whether "
                    "width moves the safe chunk length. It is also the width of "
                    "narrator's own batch (v3_served.serve_concurrency), so it "
                    "moves concurrent POSTs and VRAM pressure together. Owed: a "
                    "chars/min and VRAM reading at 12."
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
