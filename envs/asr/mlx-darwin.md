# There is no `mlx-darwin.txt` for `asr`, and there should not be one yet

If you came here looking for `envs/asr/mlx-darwin.txt`, it is missing on purpose.
`crucible install asr` on the Mac refuses by name and lists `['cuda-linux']` as
the recipes this build ships, and every `asr` manifest in `asr/` declares only a
`cuda-linux` block, so an `asr` job on `mlx-darwin` is refused with
`backend_unsupported` before it is queued.

## Why

`asr` is faster-whisper, and faster-whisper is CTranslate2. **CTranslate2 has no
Metal backend.** It supports CUDA and CPU; on Apple Silicon it builds against
Accelerate and runs on the CPU cores, and a request for `device="mps"` is a
`ValueError: unsupported device mps` (SYSTRAN/faster-whisper#515 and #911, both
still open and still true as of 2026-09). There is no version of this recipe that
puts whisper on the Mac's GPU, so a `mlx-darwin.txt` would either install
something that cannot use the accelerator or install nothing at all.

Running it on the Mac's CPU is not the compromise it looks like, either.
PHASE4-AUDIO.md section 3 already rules on the same question in the other
direction: `compute_type` is `float16` on an accelerator and `int8` on CPU, and a
transcript that quietly ran at `int8` on a CPU is a *different transcript* with
nothing in the output to say so. Crucible has no CPU backend and does not want
one by the back door.

## What the Mac needs instead

`mlx-whisper` (`mlx-community/whisper-*-mlx`). It is a different implementation
with different weights, different quantisation and its own converted repos, so it
is a second engine behind the `asr` job type rather than a second recipe behind
this one: new manifests under `asr/`, a `mlx-whisper` entry in
`ASR_BACKEND_ENGINES` (`crucible/asrmodels.py`), and a second worker script. The
wire does not change — a client still sends one file and names a model — which is
the point of the model id being the contract.

Nobody has measured `mlx-whisper` against faster-whisper on Owen's books, and
until somebody has, the two are not interchangeable at the same model id. That is
its own piece of work and it is not in phase 4.
