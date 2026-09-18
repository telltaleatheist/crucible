#!/usr/bin/env bash
# Two-point calibration for a vLLM model on THIS card (FITS-AND-THE-CARD.md §4).
#
#   scripts/calibrate-kv.sh <model-id> <util> [max-model-len]
#   KV_BYTES=<n> scripts/calibrate-kv.sh <model-id> <util> [max-model-len]
#
# Runs the engine's real argv with VLLM_LOGGING_LEVEL=DEBUG, which makes the
# memory profiler print every term of the arithmetic gpu_worker.py does:
#
#   requested      = total x util                  (worker/utils.py request_memory)
#   total_consumed = free_at_init - free_after_profile
#   non_kv         = total_consumed + transient_peak_headroom
#   available_kv   = requested - non_kv - cudagraph_estimate
#
# `GPU KV cache size: N tokens` beside `Available KV cache memory: X GiB` gives
# bytes-per-token by division -- MEASURED on this card, with no knowledge of
# layer counts, head dims or attention intervals.
#
# KV_BYTES sets --kv-cache-memory-bytes, which per vLLM's own config doc
# "(when not-None) ignores gpu_memory_utilization" and skips memory profiling
# altogether. That is the flag the fix uses, and setting it here is how the
# slope gets pinned exactly: an exact pool of N bytes reports the exact token
# count it bought.
#
# nvidia-smi is sampled at 1 Hz throughout, because `total_consumed` is a
# WHOLE-CARD delta -- a desktop that grows while vLLM profiles is charged to
# the KV pool, and that is what the 2026-09-17 failure turned out to be.
set -uo pipefail

MODEL="${1:?model id}"
UTIL="${2:?gpu-memory-utilization}"
CTX="${3:-16384}"
HOME_DIR="${CRUCIBLE_HOME:-$HOME/.crucible}"
OUT="${OUT_DIR:-${TMPDIR:-/tmp}/crucible-calib}"
mkdir -p "$OUT"
STAMP="$(date +%H%M%S)"
TAG="$MODEL.util$UTIL.ctx$CTX${KV_BYTES:+.kv$KV_BYTES}.$STAMP"
LOG="$OUT/$TAG.log"
SMI="$OUT/$TAG.smi.csv"

WEIGHTS="$HOME_DIR/models/$MODEL/cuda-linux"
[ -d "$WEIGHTS" ] || { echo "no weights at $WEIGHTS" >&2; exit 2; }

PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"

# The manifest's own engine_args, read from the manifest rather than retyped.
# The utilisation is what this script varies; everything else passes through.
mapfile -t EXTRA < <("$HOME_DIR/server/bin/python" - "$MODEL" <<'PY'
import sys
from crucible.manifests import load_manifest
for arg in load_manifest(sys.argv[1]).backends["cuda-linux"].engine_args:
    print(arg)
PY
)

ARGS=()
skip_next=0
for a in "${EXTRA[@]}"; do
  if [ "$skip_next" = 1 ]; then skip_next=0; continue; fi
  if [ "$a" = "--gpu-memory-utilization" ]; then skip_next=1; continue; fi
  ARGS+=("$a")
done

echo "== $MODEL util=$UTIL ctx=$CTX ${KV_BYTES:+kv_bytes=$KV_BYTES}"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits \
  | awk -F', ' '{printf "   nvidia-smi before: total %s MiB, used %s MiB, free %s MiB\n", $1,$2,$3}'
"$HOME_DIR/envs/llm/bin/python" -c 'import torch;f,t=torch.cuda.mem_get_info(0);print(f"   cuda mem_get_info: total {t//2**20} MiB, used {(t-f)//2**20} MiB, free {f//2**20} MiB")'

( while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | tr -d ' ')"
    sleep 1
  done ) >"$SMI" 2>/dev/null &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null' EXIT

# THE ENGINE ENV IS CRUCIBLE'S OWN, asked for rather than retyped. Without
# VLLM_WSL2_ENABLE_PIN_MEMORY the load dies with `UVA is not available` before
# it reaches the memory profiler -- a calibration that measures nothing, which
# is exactly how this harness's first run went.
eval "$("$HOME_DIR/server/bin/python" - <<'ENVDUMP'
import shlex
from crucible.engines.vllm import VllmEngine
for key, value in VllmEngine.environment(None).items():
    print(f"export {key}={shlex.quote(value)}")
ENVDUMP
)"

VLLM_LOGGING_LEVEL=DEBUG timeout 900 "$HOME_DIR/envs/llm/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$WEIGHTS" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization "$UTIL" --max-model-len "$CTX" \
  ${KV_BYTES:+--kv-cache-memory-bytes "$KV_BYTES"} \
  "${ARGS[@]}" >"$LOG" 2>&1 &
VLLM_PID=$!

# The KV verdict is printed BEFORE the server is ready, so there is nothing to
# wait for beyond it -- or beyond the process dying, which is also an answer.
for _ in $(seq 1 900); do
  grep -aq "Available KV cache memory\|reserved .* memory for\|UVA is not available" "$LOG" && break
  kill -0 "$VLLM_PID" 2>/dev/null || break
  sleep 1
done
sleep 3
kill -TERM "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null
kill $SMI_PID 2>/dev/null

echo "   log: $LOG"
for pattern in \
  "Initial free memory: [0-9.]+ GiB; Requested memory: [0-9.]+ \(util\), [0-9.]+ GiB" \
  "Memory profiling takes.*weights memory: [0-9.]+GiB\." \
  "Initial free memory [0-9.]+ GiB, reserved [0-9.]+ GiB memory for KV Cache" \
  "Model loading took [0-9.]+ GiB" \
  "Available KV cache memory: [-0-9.]+ GiB" \
  "GPU KV cache size: [0-9,]+ tokens" \
  "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" \
  "UVA is not available" ; do
  grep -aoE "$pattern" "$LOG" | tail -1 | sed 's/^/   /'
done
awk -F, 'NR>1{if($2>mx)mx=$2; if(mn==""||$2<mn)mn=$2} END{printf "   nvidia-smi used during run: min %s MiB, max %s MiB (swing %s MiB)\n", mn, mx, mx-mn}' "$SMI"
