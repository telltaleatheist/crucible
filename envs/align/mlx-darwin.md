# `envs/align/mlx-darwin.txt` exists now — and one thing it asks for is still owed

This file used to explain why there was no Mac recipe for `align`. There is one,
beside this note, since 2026-09-14. What is left here is the half of the old
demand that has **not** been met, kept in the place somebody will look for it.

## What landed

- `envs/align/mlx-darwin.txt` — the freeze of the Mac Studio's `qwen-align` env
  (the one BookForge builds, packs and publishes as
  `qwen-align-env-macos-arm64.tar.gz`), read on 2026-09-14. Against the cuda
  recipe it is the same set minus the 19 `nvidia-*`/`cuda-*` wheels and
  `triton`, with `platformdirs` one patch newer. Same `qwen-asr==0.0.6`, same
  `torch==2.14.0` (the macOS arm64 wheel IS the Metal build), same
  `transformers==4.57.6`.
- `mlx-darwin: "qwen3-forced-aligner"` in `ALIGN_BACKEND_ENGINES` — the SAME
  engine on both backends, which is the whole difference between this job type
  and `asr`.
- `[backends.mlx-darwin]` on `align/qwen3-aligner.toml`: the identical repo and
  revision, `dtype = "bfloat16"`, and a **measured** `memory_bytes_estimate` —
  5,885,296,640 B, the high-water of `torch.mps.driver_allocated_memory()` over
  three back-to-back 300-second chunks on the M1 Ultra on 2026-09-14. It is not
  the cuda figure and it is not arithmetic: three of the four things cuda's
  declared 1.5 GiB covers (CUDA context, cuBLAS and cuDNN workspaces) do not
  exist on Metal, and the real number turned out to be LARGER than the cuda
  one, because torch's MPS caching allocator keeps what it takes.
- `crucible/jobs/align/__init__.py` picks the device per backend —
  `cuda-linux` → `cuda`, `mlx-darwin` → `mps` — with no default, so a backend
  nobody has decided about is a refusal and never a `cuda` handed to a Mac. The
  dtype still comes off the manifest; only the device is the backend's.
- An `align/mlx-darwin` row in `.github/workflows/envpacks.yml`.

The wire did not change. A client still names a model and sends chunks.

## What was measured, and what those measurements are about

**Speed, twice, and they agree.** BookForge measured **97x realtime warm on MPS
in bfloat16** — 33 s cold, model load included, for 95 s of audio — on
2026-09-08 (`electron/components/qwen-align-env.ts:8-9`). Crucible measured
**about 77x warm** on 2026-09-14 in the same env: 900 s of audio (three 300 s
chunks, the ceiling this job type accepts) aligned in 11.72 s, with the cold
first chunk at 5.39 s for 300 s. The bake-off that chose this aligner over
WhisperX got 229x on the 3090 Ti in WSL2, so the Mac is slower and nowhere near
slow enough to matter.

**Memory, once, and it is in the manifest.** 5,885,296,640 B — see
`align/qwen3-aligner.toml`, which carries the run, the method and the three
per-chunk readings that show it is a ceiling rather than a leak.

What all of that settles is that the aligner runs on Metal, at a usable rate, in
a known footprint, in the env this recipe freezes.

## What is STILL owed, and it is the important half

**The timestamp comparison.** Speed says nothing about agreement. `bfloat16` on
MPS is a different numerical path from `bfloat16` on CUDA, and several torch
operators fall back to `float32` or to the CPU there; a timestamp that moved
because of that looks exactly like a timestamp that did not.

One person, one chapter, both machines:

1. Align a chapter on the PC (`cuda-linux`) and the same chapter on the Mac
   (`mlx-darwin`), same model id, same revision, same text.
2. **Compare the timestamps** — not eyeball the output, and not merely check
   that it ran. Within the tolerance BookForge's own coverage report already
   uses is the bar.
3. Write the result into `docs/PHASE15-HOST.md` section 7c, whichever way it
   goes. A disagreement is a finding about MPS and belongs in the manifest's
   comment; an agreement is what lets a reader stop wondering.

The memory figure is no longer owed — it was measured on 2026-09-14 and the
manifest shows the run. The one number still worth re-reading is what a real
book does: every chunk above was the 300 s worst case, and narrator's chunks are
nearer 90 s, so a real run should sit under the manifest's figure rather than
over it.
