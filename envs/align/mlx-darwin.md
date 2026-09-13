# There is no `mlx-darwin.txt` for `align`, and the reason is not the same as `asr`'s

If you came here looking for `envs/align/mlx-darwin.txt`, it is missing on
purpose. `crucible install align` on the Mac refuses by name and lists
`['cuda-linux']` as the recipes this build ships, `align/qwen3-aligner.toml`
declares only a `cuda-linux` block, and an `align` job on `mlx-darwin` is refused
with `backend_unsupported` before it is queued.

## Why, and how this differs from `asr`

`envs/asr/mlx-darwin.md` is missing because it **cannot exist**: faster-whisper is
CTranslate2, CTranslate2 has no Metal backend, and `device="mps"` is a
`ValueError`. There is no version of that recipe that puts whisper on the Mac's
GPU.

This one is different, and the difference matters. Qwen3-ForcedAligner-0.6B is a
plain torch model loaded through `qwen_asr.Qwen3ForcedAligner.from_pretrained`,
torch has an MPS backend, and BookForge's own aligner already accepts `mps` as a
device (`python/narrator/align/aligner.py` picks `float32` only on `cpu`, and
`bfloat16` on everything else, `mps` included). So a Mac recipe is *possible*.

It is missing because **nobody has measured it**, and this repo does not ship a
recipe as a way of asserting a result it does not have. Specifically:

- The bake-off that chose this aligner over WhisperX — 229x realtime and 51 of 61
  cues exact, against 18x and 39 — was run in WSL2 on the 3090 Ti on 2026-09-08.
  Nothing in it says anything about Metal.
- `bfloat16` on MPS is a different numerical path from `bfloat16` on CUDA, and
  several torch operators fall back to `float32` or to the CPU there. A timestamp
  that moved because of that would look exactly like a timestamp that did not.
- The CUDA recipe beside this file pins about 25 `nvidia-*` wheels. A Mac recipe
  is not that file with the CUDA lines removed; it is a resolve of its own, on a
  Mac, with a torch that has to be a working one for this model.

## What it would take

One person, one Mac, one book. Install `qwen-asr` and a Metal torch into a venv,
run `crucible install align --force`, align a chapter that has already been
aligned on the PC, and **compare the timestamps** — not eyeball the output, and
not simply check that it ran. If they agree within the tolerance BookForge's own
coverage report already uses, write `mlx-darwin.txt` from that env's `pip
freeze`, add an `[backends.mlx-darwin]` block to `align/qwen3-aligner.toml` with
its own measured `memory_bytes_estimate`, and add `mlx-darwin` to
`ALIGN_BACKEND_ENGINES` in `crucible/alignmodels.py`. The wire does not change; a
client still names a model and sends chunks.

Until then the honest answer on a Mac is `backend_unsupported`, and the honest
place to put the reason is here, where somebody looking for the missing file will
find it.
