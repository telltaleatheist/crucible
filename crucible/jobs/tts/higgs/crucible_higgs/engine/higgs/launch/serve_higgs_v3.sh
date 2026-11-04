#!/bin/bash
# Launch Higgs Audio v3 (4B) under vllm-omni.
#
# THIS FILE IS NARRATOR'S OWN LAUNCHER AND SHIPS INSIDE THE PACKAGE
# (`narrator/engine/higgs/launch/`, declared in python/pyproject.toml's
# [tool.setuptools.package-data], so a `pip install` from git carries it).
# `v3_served.HiggsV3ServedBackend` resolves it through importlib.resources when
# the caller names neither NARRATOR_HIGGS3_URL nor NARRATOR_HIGGS3_SERVE_SCRIPT.
# That is not a fallback: HOW NARRATOR LAUNCHES ITS SERVER IS NARRATOR'S OWN
# DEFINITION, and the env var is an OPERATOR'S OVERRIDE of it. Before
# 2026-09-13 the only copy lived in BookForge's electron/scripts/higgs/, which
# meant every other client of narrator - Crucible first - had to be handed a
# path into a BookForge checkout or narrator refused to start at all.
#
# electron/scripts/higgs/serve_higgs_v3.sh is kept BYTE-IDENTICAL to this file
# by tools/test-higgs-engine.js. That copy belongs to BookForge's LEGACY local
# render path (docs/CRUCIBLE_ROLLOUT_PLAN.md item 2.4: the local narrator spawn
# is a dated stopgap, deleted after Owen's in-app pass) and dies with it; THIS
# one is canonical.
#
# PROVENANCE. Transcribed from
#   E:\training\_campaigns\2026-09-01-cod-full-rebuild\higgs\serve_v3.sh
# (example-pc, WSL Ubuntu, RTX 3090 Ti, 2026-09-04), which is the script every
# measurement in electron/data/higgs-models.json was taken against. What differs
# from the original, and why each one is a fix rather than drift:
#   1. The env prefix is a REQUIRED parameter ($HIGGS_ENV) instead of a
#      hardcoded /home/<user> path, so this ships. It had a default of
#      `$HOME/anaconda3/envs/higgs3` until 2026-09-13; that is ONE MACHINE'S
#      CONDA LAYOUT, and a caller that forgot the variable got a server started
#      out of a directory nobody named - or, on a machine without that env, a
#      `vllm-omni: No such file` three lines from the end of a launch. Both
#      callers state it: BookForge from the catalog (`higgsSpawnEnv`), Crucible
#      from the tts env's own venv root.
#   2. The model is resolved from $HIGGS_MODEL_DIR when set, so a merged
#      fine-tune can be served without editing the script. narrator EXPORTS it
#      per voice (python/narrator/engine/higgs/v3_served.py `_launch_exports`).
#   3. Memory, concurrency and context length are passed PER STAGE through
#      --stage-overrides, not as global flags. vllm-omni applies a global
#      --gpu-memory-utilization / --max-num-seqs / --max-model-len to EVERY stage
#      (vllm_omni/config/omni_config.py `_stage_cli_overrides`: a field of
#      OmniEngineArgs is a "global stage field"), and this server is TWO stages -
#      the talker (stage 0) and the codec decoder (stage 1). The campaign's
#      `--gpu-memory-utilization 0.60` therefore reserved 0.60 of the card TWICE:
#      measured 24.2 GB of 24.5 GB in use on 2026-09-05. The deploy profile
#      (vllm_omni/deploy/higgs_multimodal_qwen3.yaml) intends 0.60 for the talker
#      and 0.25 for the codec, and that is what the two variables below default
#      to. Likewise `--max-model-len 8192` was clamping the codec stage, whose
#      profile value is 65536.
#   4. $HIGGS_DEPLOY_CONFIG selects vllm-omni's deploy profile, and it DEFAULTS
#      TO THE SIBLING FILE, $(dirname "$0")/higgs_default_frames7500.yaml -
#      which ships beside this script in the package and is what the installer
#      copies into <env>/bin/ beside the legacy copy. It defaulted to EMPTY
#      until 2026-09-13, i.e. to vllm-omni's own auto-discovered profile, whose
#      stage-0 max_tokens is 2048 frames = 81.92 s - so a caller that did not
#      know to set this variable truncated every long chunk and nothing said
#      so. BookForge sets it explicitly from the catalog either way, so this
#      default changes nothing for it; it is what makes the packaged launcher
#      correct for a caller who has only read this file.
#      CERTIFICATES BIND TO THAT PROFILE BY SHA256. The shipped bytes are
#      24d288f193eaa8c5c10387d890b648949b88e8d357a3779e8c9487f9d38c7481, and
#      .gitattributes pins the file to LF for exactly that reason: a CRLF
#      checkout parses identically as YAML and is no longer the file any cap
#      certificate was measured against.
#      THAT PROFILE EXISTS BECAUSE THE SERVED SPEECH ENDPOINT IGNORES A
#      PER-REQUEST max_tokens. MEASURED 2026-09-05 (example-pc, vllm-omni 0.28.0):
#      stage 0's `default_sampling_params.max_tokens` is a HARD CEILING on every
#      render, and the auto-discovered profile sets it to 2048 frames = 81.92 s,
#      so any chunk needing more audio than that is cut mid-sentence and no
#      request parameter can raise it. The shipped profile is byte-identical to
#      vllm-omni's own `higgs_multimodal_qwen3.yaml` EXCEPT max_tokens 7500
#      (300 s). Unset, vllm-omni auto-discovers its own file again and the
#      81.92 s ceiling is back - which is why this is set rather than left to a
#      default. (The sibling `higgs_multimodal_qwen3_low_latency.yaml` is a
#      different question - CUDA graphs on the talker - and is still uncertified.)
#   5. The attention backend is forced PER STAGE in --stage-overrides as well as
#      by the global --attention-backend flag. MEASURED (training, 2026-09-05):
#      when a deploy profile is passed EXPLICITLY, a stage-level
#      `attention_backend: FLASHINFER` in that profile can beat the global flag -
#      and FlashInfer cannot JIT on this box (the CUDA 13 nvcc from the pip wheel
#      rejects its bundled CCCL headers), so the global flag alone is not enough
#      to keep it out. Passing it in the stage override says it at the level the
#      profile says it at.
# The levers are documented in the campaign's HIGGS_V3_LEVERS.md.
#
# BookForge's installer copies its own (byte-identical) copy into
# <env>/bin/serve_higgs_v3.sh and chmod +x's it;
# the doctor checks that the copy there is byte-identical to the shipped one.
# Keeping it inside the env is what makes the stack self-contained: nothing at
# run time reads E:\training.
set -euo pipefail

# REQUIRED, NO DEFAULT - see note 1 in the header.
if [ -z "${HIGGS_ENV:-}" ]; then
  echo "HIGGS_ENV is not set. It is the environment prefix this server runs out of:" >&2
  echo "CUDA_HOME, PATH, LD_LIBRARY_PATH and \$HIGGS_ENV/bin/vllm-omni are all built" >&2
  echo "from it. There is no default - the old one was one machine's conda layout." >&2
  exit 5
fi
HIGGS_PORT="${HIGGS_PORT:-8095}"
HIGGS_HOST="${HIGGS_HOST:-127.0.0.1}"
# Stage 0, the talker: weights + KV cache. Stage 1, the codec decoder: no KV
# cache. Both are FRACTIONS OF THE WHOLE CARD, and they add - and the sum must
# leave room for three CUDA contexts.
#
# THE TALKER'S FRACTION IS ITS KV CACHE BUDGET ON TOP OF THE WEIGHTS, not a cap
# on the stage. MEASURED 2026-09-05 (example-pc, RTX 3090 Ti 24.5 GB, vllm-omni
# 0.28.0) at 0.35, from the server's own log: "Desired GPU memory utilization is
# (0.35, 8.4 GiB). Actual usage is 7.72 GiB", "Available KV cache memory:
# 8.4 GiB", "GPU KV cache size: 61,120 tokens" - so stage 0 holds 7.72 GiB of
# weights AND 8.4 GiB of cache. Reading the fraction as a ceiling is what made
# the earlier pairs overcommit the card:
#
#   0.60 + 0.25   24,274 MiB in use, the WDDM driver paging into shared system
#                 RAM, the render falling from ~8 to ~2 chunks/min, 2 GB free.
#   0.55 + 0.15   24.0 GB in use - still paging.
#   0.35 + 0.10   18.7-19.2 GB in use, and IDENTICAL THROUGHPUT at 16 concurrent:
#                 11,387-11,584 chars/min across three runs. Shipped.
#
# At 0.35 + 0.10, 32 concurrent filled the card and stalled, which is the other
# half of why HIGGS_MAX_NUM_SEQS ships at 16.
HIGGS_GPU_MEM_UTIL="${HIGGS_GPU_MEM_UTIL:-0.35}"
HIGGS_CODEC_GPU_MEM_UTIL="${HIGGS_CODEC_GPU_MEM_UTIL:-0.10}"
HIGGS_MAX_MODEL_LEN="${HIGGS_MAX_MODEL_LEN:-8192}"
HIGGS_MAX_NUM_SEQS="${HIGGS_MAX_NUM_SEQS:-16}"
# THE CERTIFIED frames-7500 PROFILE, which ships beside this file - see note 4.
#
# `${VAR-...}` AND NOT `${VAR:-...}`, deliberately: UNSET means "nobody said, so
# take the certified profile", while SET-BUT-EMPTY means "vllm-omni's own
# auto-discovered profile, chosen on purpose". That distinction is the only way
# the auto profile stays reachable now that it is no longer what silence means,
# and BookForge's catalog needs it - `deployConfig: null` there is a decision
# (higgsSpawnEnv emits the empty string for it) and a missing key is refused.
HIGGS_DEPLOY_CONFIG="${HIGGS_DEPLOY_CONFIG-$(dirname "$0")/higgs_default_frames7500.yaml}"

# WHICH STACK THE CALLER THINKS IT IS STARTING, asserted rather than assumed.
#
# BookForge sets HIGGS_STACK from the catalog's `serving.stack` on every door,
# and narrator refuses to render without it (served_common.serving_stack) because
# the two stacks want different request bodies, different frame caps and
# different sampling. THIS is the other half of that contract: a job configured
# for SGLang-Omni that somehow reached the vllm-omni launcher would come up on a
# server whose requests the client is not building — which is not a crash, it is
# a book rendered at whatever the dropped fields defaulted to. Refuse in two
# seconds instead of 297.
#
# Unset is accepted: this script is also run by hand, and by an operator who has
# chosen it by typing its name.
if [ -n "${HIGGS_STACK:-}" ] && [ "$HIGGS_STACK" != "vllm-omni" ]; then
  echo "HIGGS_STACK is '$HIGGS_STACK' but this is the vllm-omni launcher." >&2
  echo "Start serve_higgs_sgl.sh for the sglang-omni stack, or fix serving.stack in" >&2
  echo "electron/data/higgs-models.json — the client and the server must agree." >&2
  exit 6
fi

# The CUDA 13 toolkit that ships INSIDE the pip wheel. vllm-omni's stack is built
# against it and the system CUDA (if any) is the wrong version.
export CUDA_HOME="$HIGGS_ENV/lib/python3.11/site-packages/nvidia/cu13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$HIGGS_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# FlashInfer is unavailable on sm_86 with torch 2.13+cu130: it JIT-builds its
# attention kernels with nvcc, and its bundled CCCL headers reject the CUDA 13
# nvcc from the pip wheel. These three lines route around it entirely — the
# torch-native sampler plus vLLM's prebuilt FA2. This is SPEED ONLY; correctness
# is unaffected. The flashinfer-jit-cache cu130/sm_86 wheel from
# https://flashinfer.ai/whl/cu130/torch2.13/ is the route if v3 speed matters.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_DISABLE_FLASHINFER_PREFILL=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

# A fine-tune wins; otherwise serve the base snapshot out of the HF cache.
#
# HIGGS_MODEL_DIR IS A MERGED CHECKPOINT DIRECTORY (~8.5 GB), by convention at
# /home/<user>/higgs-models/<voice>/. It is not a LoRA and there is no flag here
# that would take one: vllm-omni cannot load an adapter at runtime — `vllm-omni
# serve` has no adapter options and the higgs_audio_v3 talker class does not
# implement SupportsLoRA — so a LoRA is merged into a full checkpoint before it
# can serve at all. That is also why changing voice means restarting this script
# (~55 s warm / ~300 s cold) rather than sending the server a message.
#
# NO FALLBACK ON A SET-BUT-MISSING DIR: if the caller named a model dir and it is
# not there, that is a wrong render waiting to happen — serving the base instead
# would produce a completely different narrator and report success.
if [ -n "${HIGGS_MODEL_DIR:-}" ]; then
  if [ ! -d "$HIGGS_MODEL_DIR" ]; then
    echo "HIGGS_MODEL_DIR is set to '$HIGGS_MODEL_DIR' but that directory does not exist." >&2
    echo "Refusing to serve the base model in its place — it is a different speaker." >&2
    exit 2
  fi
  MODEL="$HIGGS_MODEL_DIR"
else
  MODEL=$(ls -d "$HOME"/.cache/huggingface/hub/models--bosonai--higgs-audio-v3-tts-4b/snapshots/*/ 2>/dev/null | head -1)
  if [ -z "$MODEL" ]; then
    echo "Higgs v3 weights are not in the HuggingFace cache and no HIGGS_MODEL_DIR was given." >&2
    exit 3
  fi
fi

# Refuse a value that is not a number before vllm-omni turns it into a traceback
# three minutes in. (bash has no floats; the pattern is the check.)
for pair in "HIGGS_GPU_MEM_UTIL=$HIGGS_GPU_MEM_UTIL" "HIGGS_CODEC_GPU_MEM_UTIL=$HIGGS_CODEC_GPU_MEM_UTIL"; do
  case "${pair#*=}" in
    0.[0-9]*|1|1.0) ;;
    *) echo "${pair%%=*} must be a fraction in (0, 1]; got '${pair#*=}'" >&2; exit 4 ;;
  esac
done
case "$HIGGS_MAX_NUM_SEQS" in ''|*[!0-9]*|0) echo "HIGGS_MAX_NUM_SEQS must be a positive integer; got '$HIGGS_MAX_NUM_SEQS'" >&2; exit 4 ;; esac
case "$HIGGS_MAX_MODEL_LEN" in ''|*[!0-9]*|0) echo "HIGGS_MAX_MODEL_LEN must be a positive integer; got '$HIGGS_MAX_MODEL_LEN'" >&2; exit 4 ;; esac

# `attention_backend` is stated on stage 0 as well as globally: see note 5 in the
# header. A deploy profile passed with --deploy-config carries its own
# stage-level attention_backend, and that beat the global --attention-backend
# flag when the profile was given explicitly (training, 2026-09-05) - so the
# override has to say it at the same level the profile does.
STAGE_OVERRIDES="{\"0\": {\"gpu_memory_utilization\": $HIGGS_GPU_MEM_UTIL, \"max_num_seqs\": $HIGGS_MAX_NUM_SEQS, \"max_model_len\": $HIGGS_MAX_MODEL_LEN, \"attention_backend\": \"$VLLM_ATTENTION_BACKEND\"}, \"1\": {\"gpu_memory_utilization\": $HIGGS_CODEC_GPU_MEM_UTIL, \"max_num_seqs\": $HIGGS_MAX_NUM_SEQS}}"

# A deploy profile is a FILE NAME or a full path, never a bare name. MEASURED
# (training, 2026-09-05, vllm-omni 0.28.0): `--deploy-config
# higgs_multimodal_qwen3_low_latency` fails at startup with "Deploy config not
# found" - config_factory._load_user_deploy_config joins a bare name to the
# deploy dir without appending .yaml. `higgs_multimodal_qwen3_low_latency.yaml`
# resolves. Refused here by name rather than passed through to a crash that
# costs a 55-297 s launch to read.
DEPLOY_ARGS=()
if [ -n "$HIGGS_DEPLOY_CONFIG" ]; then
  case "$HIGGS_DEPLOY_CONFIG" in
    *.yaml|*.yml|*/*) ;;
    *) echo "HIGGS_DEPLOY_CONFIG='$HIGGS_DEPLOY_CONFIG' is a bare profile name; vllm-omni resolves only a file name (append .yaml) or a full path." >&2; exit 4 ;;
  esac
  DEPLOY_ARGS=(--deploy-config "$HIGGS_DEPLOY_CONFIG")
fi

echo "MODEL=$MODEL"
echo "BIND=$HIGGS_HOST:$HIGGS_PORT"
echo "STAGE_OVERRIDES=$STAGE_OVERRIDES"
echo "DEPLOY_CONFIG=${HIGGS_DEPLOY_CONFIG:-(vllm-omni default profile)}"

exec "$HIGGS_ENV/bin/vllm-omni" serve "$MODEL" \
  --served-model-name higgs-v3 \
  --host "$HIGGS_HOST" --port "$HIGGS_PORT" \
  --trust-remote-code \
  --stage-overrides "$STAGE_OVERRIDES" \
  --attention-backend "$VLLM_ATTENTION_BACKEND" \
  "${DEPLOY_ARGS[@]}" \
  --omni
