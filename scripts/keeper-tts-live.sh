#!/usr/bin/env bash
# Live keeper for the `tts` render door: a real server, a real narrator, a real
# voice, and real FLACs on disk.
#
# This is the script that discharges the owed item in PHASE3-TTS.md section 10.
# Everything in phase 3b was built against `tests/fake_narrator.py` because both
# of Owen's cards were busy the night it was written, so no voice manifest
# carries a MEASURED memory figure — every one of them says `estimate_basis =
# "declared"`. This run is what turns that into a measurement, and it prints the
# exact lines to paste back into the manifest.
#
# LOCAL MODE (default). Starts a throwaway server on a free port against the
# operator's real ~/.crucible, because the tts env and the voice weights live
# there and are gigabytes. It does not install either — that is minutes of
# downloading and not a keeper's job — but it refuses BY NAME if they are
# missing rather than skipping.
#
#   ./scripts/keeper-tts-live.sh
#   CRUCIBLE_VOICE=mistborn ./scripts/keeper-tts-live.sh
#
# REMOTE MODE. Set both CRUCIBLE_URL and CRUCIBLE_TOKEN and the keeper drives a
# server already running somewhere else, starting nothing of its own — the shape
# that matters for the apps, a Windows client driving a GPU on another machine.
# In remote mode the card cannot be watched, so the memory measurement is
# skipped and said to be skipped; everything else still runs. Setting one of the
# two variables and not the other is a refusal, never a silent fall back.
#
# What it proves, end to end:
#   1. the server is reachable and says what it is
#   2. the voice is installed and loadable there, pinned to a revision, and
#      /info's tts capability carries exactly the same rows as /voices
#   3. nothing is resident to begin with
#   4. a render LOADS its own voice — the one asymmetry with `llm` — streaming
#      `warming` and ending `done`
#   5. one `<index>.flac` per chunk, each a real FLAC at the voice's own sample
#      rate, mono, and non-trivially long
#   6. a `chunk` event per row carrying seconds, chars and chars_per_sec
#   7. the provenance sidecar names the merge that rendered it
#   8. a second render reuses the resident voice and does not restart narrator
#   9. unload-voice frees the engine and nothing is resident afterwards
#  10. (local mode only) the card comes back to where it started, and the peak
#      while rendering is printed as the manifest's measured estimate
#
# Exits 0 only if every check passed. Trust the exit code.
#
# THE CARD IS NOT TAKEN WITHOUT ASKING. This loads a model. Owen's standing rule
# is that Crucible never starts GPU work while somebody else is on the card, so
# the keeper reads /v1/accelerator first and refuses by name if anything that is
# not this server's own is holding more than the desktop allowance.

set -euo pipefail

VOICE="${CRUCIBLE_VOICE:-deathstalker}"
LOAD_TIMEOUT="${CRUCIBLE_LOAD_TIMEOUT:-1200}"
RENDER_TIMEOUT="${CRUCIBLE_RENDER_TIMEOUT:-900}"

PASSED=0
FAILED=0
SERVER_PID=""
ROOT=""

log()  { printf '  %s\n' "$*"; }
ok()   { PASSED=$((PASSED + 1)); printf 'ok    %s\n' "$*"; }
bad()  { FAILED=$((FAILED + 1)); printf 'FAIL  %s\n' "$*" >&2; }
die()  { printf 'keeper: %s\n' "$*" >&2; exit 2; }

cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    # SIGTERM only. The server's own shutdown stops the resident engine the same
    # way, and narrator's teardown stops SGLang under it. A SIGKILL here would
    # leave a CUDA process wedged in WSL2 until Windows reboots.
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  [ -n "$ROOT" ] && [ -d "$ROOT" ] && rm -rf "$ROOT"
  return 0
}
trap cleanup EXIT

for tool in curl python3; do
  command -v "$tool" >/dev/null || die "no $tool on PATH"
done

# ------------------------------------------------------------------ the server

REMOTE=0
if [ -n "${CRUCIBLE_URL:-}" ] || [ -n "${CRUCIBLE_TOKEN:-}" ]; then
  [ -n "${CRUCIBLE_URL:-}" ] && [ -n "${CRUCIBLE_TOKEN:-}" ] \
    || die "remote mode needs BOTH CRUCIBLE_URL and CRUCIBLE_TOKEN; one without the other is ambiguous and this never guesses"
  REMOTE=1
  BASE="${CRUCIBLE_URL%/}"
  case "$BASE" in */v1) ;; *) BASE="$BASE/v1" ;; esac
  TOKEN="$CRUCIBLE_TOKEN"
  log "remote mode: $BASE"
else
  command -v crucible >/dev/null || die "no crucible on PATH (or set CRUCIBLE_URL and CRUCIBLE_TOKEN for remote mode)"
  HOME_DIR="${CRUCIBLE_HOME:-$HOME/.crucible}"
  [ -f "$HOME_DIR/config.toml" ] || die "no config at $HOME_DIR/config.toml — run \`crucible init\` first"
  PORT="$(python3 -c 'import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
  TOKEN="$(CRUCIBLE_HOME="$HOME_DIR" crucible token --show)"
  BASE="http://127.0.0.1:$PORT/v1"
  ROOT="$(mktemp -d)"
  log "local mode: a throwaway server on $PORT against $HOME_DIR"
  CRUCIBLE_HOME="$HOME_DIR" crucible serve --host 127.0.0.1 --port "$PORT" \
    >"$ROOT/serve.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    curl -fsS "$BASE/ping" >/dev/null 2>&1 && break
    sleep 0.5
  done
fi

WORK="${ROOT:-$(mktemp -d)}"
mkdir -p "$WORK/out"
AUTH=(-H "Authorization: Bearer $TOKEN" -H "X-Crucible-Api: 1")

get()  { curl -fsS "${AUTH[@]}" "$BASE/$1"; }
post() { curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' --data-binary "@$2" "$BASE/$1"; }

jq_py() { python3 -c "$1" ; }

# ------------------------------------------------------------------- 1. ping

if get ping >/dev/null 2>&1 || curl -fsS "$BASE/ping" >/dev/null; then
  NAME="$(curl -fsS "$BASE/ping" | python3 -c 'import json,sys; print(json.load(sys.stdin)["name"])')"
  ok "the server answers /ping and calls itself $NAME"
else
  bad "the server did not answer /ping at $BASE"
  exit 1
fi

# ----------------------------------------------------- 2. the voice is loadable

get voices >"$WORK/voices.json" || { bad "GET /voices failed"; exit 1; }
get info   >"$WORK/info.json"   || { bad "GET /info failed"; exit 1; }

python3 - "$WORK/voices.json" "$WORK/info.json" "$VOICE" <<'PY' >"$WORK/voice.txt" || { bad "the voice is not usable here (above)"; exit 1; }
import json, sys
voices = json.load(open(sys.argv[1]))
info = json.load(open(sys.argv[2]))
wanted = sys.argv[3]

row = next((v for v in voices if v["id"] == wanted), None)
if row is None:
    sys.exit(f"keeper: this server does not know the voice {wanted!r}; it has "
             f"{[v['id'] for v in voices]}")
if not row["loadable"]:
    sys.exit(f"keeper: {wanted} is not loadable here: {row['reason']}")

caps = {c["job_type"]: c["models"] for c in info["capabilities"]}
if "tts" not in caps:
    sys.exit("keeper: /info has no tts capability; is [jobs] enable_tts on?")
if caps["tts"] != voices:
    sys.exit("keeper: /info's tts rows are not /voices' rows verbatim — one "
             "voice, one description (PHASE3-TTS.md section 2)")
print(row["revision"])
print(row["sample_rate"])
print(row.get("estimate_basis"))
print(row["memory_bytes_estimate"])
PY

REVISION="$(sed -n 1p "$WORK/voice.txt")"
SAMPLE_RATE="$(sed -n 2p "$WORK/voice.txt")"
BASIS="$(sed -n 3p "$WORK/voice.txt")"
DECLARED="$(sed -n 4p "$WORK/voice.txt")"
ok "$VOICE is loadable, pinned at ${REVISION:0:12}, and /info's tts rows are /voices' rows"
log "its estimate is $DECLARED bytes, basis: $BASIS"

# ------------------------------------------------- 3. the card, before anything

BEFORE=""
if [ "$REMOTE" = "0" ]; then
  get accelerator >"$WORK/accel-before.json" || { bad "GET /accelerator failed"; exit 1; }
  python3 - "$WORK/accel-before.json" <<'PY' || { bad "somebody else is on the card (above)"; exit 1; }
import json, sys
state = json.load(open(sys.argv[1]))
foreign = [h for h in state["holders"] if not h["owned_by_crucible"]]
stray = state.get("unattributed_bytes") or 0
if foreign or stray > (1 << 30):
    sys.exit("keeper: this card is not free. holders=%s unattributed=%s. "
             "Crucible never evicts anybody and neither does this keeper."
             % (foreign, stray))
PY
  BEFORE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["used_bytes"])' "$WORK/accel-before.json")"
  ok "the card is free; it holds $BEFORE bytes before we start"
fi

get health | python3 -c '
import json, sys
h = json.load(sys.stdin)
assert h["resident_models"] == [], h
assert h["resident_kind"] is None, h
' && ok "nothing is resident to begin with" || bad "something was already resident"

# ------------------------------------------ 4-7. a render, which loads its voice

cat >"$WORK/render.json" <<JSON
{"type": "tts", "model": "$VOICE", "params": {"language": "en", "take": 0, "chunks": [
  {"index": 41, "text": "He had been walking for some time, and the road did not appear to end."},
  {"index": 42, "text": "By evening he had stopped counting the mileposts altogether."}
]}}
JSON

JOB="$(post jobs "$WORK/render.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')"
log "render job $JOB"

# The card is sampled while the render runs, because the peak is what the guard
# has to be able to give and it is not visible once the run is over.
PEAK_FILE="$WORK/peak"
if [ "$REMOTE" = "0" ] && command -v nvidia-smi >/dev/null; then
  ( best=0
    while kill -0 $$ 2>/dev/null; do
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
      [ -n "$used" ] && [ "$used" -gt "$best" ] && best="$used"
      echo "$best" >"$PEAK_FILE"
      sleep 2
    done ) &
  SAMPLER=$!
fi

curl -fsS --max-time "$RENDER_TIMEOUT" "${AUTH[@]}" \
  "$BASE/jobs/$JOB/events" >"$WORK/events.sse" || { bad "the event stream failed"; exit 1; }
[ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null || true

python3 - "$WORK/events.sse" <<'PY' >"$WORK/render.txt" || { bad "the render did not do what it promised (above)"; exit 1; }
import json, sys

kinds, events = [], []
kind = None
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.rstrip("\n")
    if line.startswith("event: "):
        kind = line[len("event: "):].strip()
    elif line.startswith("data: ") and kind:
        events.append((kind, json.loads(line[len("data: "):])))
        kinds.append(kind)

terminal = events[-1]
if terminal[0] != "done":
    sys.exit(f"keeper: the render ended {terminal[0]}: {terminal[1]}")
if "warming" not in kinds:
    sys.exit("keeper: the render never said `warming` — a render is supposed to "
             "load its own voice (PHASE3-TTS.md section 6)")

chunks = [d for k, d in events if k == "chunk"]
if len(chunks) != 2:
    sys.exit(f"keeper: expected 2 chunk events, got {len(chunks)}")
for row in chunks:
    for field in ("index", "seconds", "chars", "chars_per_sec", "take"):
        if field not in row:
            sys.exit(f"keeper: a chunk event has no {field}: {row}")
    if not row["seconds"] > 0:
        sys.exit(f"keeper: chunk {row['index']} reported {row['seconds']}s")
    print(f"chunk {row['index']}: {row['seconds']:.2f}s, {row['chars']} chars, "
          f"{row['chars_per_sec']:.1f} chars/s, capped={row['capped']}")

artifacts = sorted(d["name"] for k, d in events if k == "artifact")
if artifacts != ["41.flac", "42.flac"]:
    sys.exit(f"keeper: artifacts were {artifacts}, not one flac per chunk")
PY
ok "the render loaded its voice, streamed warming, and ended done"
sed 's/^/  /' "$WORK/render.txt"
ok "a chunk event per row, each carrying its own measurement"

for name in 41.flac 42.flac; do
  curl -fsS "${AUTH[@]}" "$BASE/jobs/$JOB/artifacts/$name" >"$WORK/out/$name"
  curl -fsS "${AUTH[@]}" "$BASE/jobs/$JOB/artifacts/$name.provenance.json" \
    >"$WORK/out/$name.provenance.json"
done

python3 - "$WORK/out" "$SAMPLE_RATE" "$VOICE" "$REVISION" <<'PY' || { bad "the artifacts are not what they claim (above)"; exit 1; }
import json, struct, sys
from pathlib import Path

out, rate, voice, revision = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
for name in ("41.flac", "42.flac"):
    raw = (out / name).read_bytes()
    if raw[:4] != b"fLaC":
        sys.exit(f"keeper: {name} does not start with fLaC")
    if len(raw) <= 1024:
        sys.exit(f"keeper: {name} is {len(raw)} bytes — BookForge's resume test is "
                 "'exists and exceeds 1024 bytes', so this would read as a finished "
                 "chunk and the sentence would be silently lost from the book")
    # STREAMINFO is the first metadata block: 4 bytes magic, 4 header, then 18
    # bytes whose bits 80..99 are the sample rate and 100..102 the channel count.
    body = raw[8:8 + 18]
    bits = int.from_bytes(body[10:14], "big")
    sample_rate = bits >> 12
    channels = ((bits >> 9) & 0x7) + 1
    if sample_rate != rate:
        sys.exit(f"keeper: {name} is {sample_rate} Hz, the voice says {rate}")
    if channels != 1:
        sys.exit(f"keeper: {name} has {channels} channels, not mono")

    sidecar = json.loads((out / f"{name}.provenance.json").read_text())
    model = sidecar["model"]
    if model["id"] != voice or model["revision"] != revision:
        sys.exit(f"keeper: {name}'s sidecar names {model}, not {voice}@{revision}")
print("both flacs are real, mono, at the voice's own rate, and named their merge")
PY
ok "the bytes are real FLACs, mono at $SAMPLE_RATE Hz, and each names the merge that made it"

# ------------------------------------------- 8. a second render does not reload

JOB2="$(post jobs "$WORK/render.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')"
curl -fsS --max-time "$RENDER_TIMEOUT" "${AUTH[@]}" "$BASE/jobs/$JOB2/events" >"$WORK/events2.sse"
if grep -q '^event: warming' "$WORK/events2.sse"; then
  bad "the second render said warming — the voice should already have been resident"
else
  ok "a second render reused the resident voice and did not restart narrator"
fi

# ------------------------------------------------------------- 9. unload

if [ "$REMOTE" = "0" ] && [ -f "$PEAK_FILE" ]; then
  PEAK_MIB="$(cat "$PEAK_FILE")"
fi

printf '{"type": "unload-voice", "model": "%s"}\n' "$VOICE" >"$WORK/unload.json"
JOB3="$(post jobs "$WORK/unload.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')"
curl -fsS --max-time 300 "${AUTH[@]}" "$BASE/jobs/$JOB3/events" >/dev/null
get health | python3 -c '
import json, sys
h = json.load(sys.stdin)
assert h["resident_models"] == [], h
assert h["resident_kind"] is None, h
' && ok "unload-voice freed the engine and nothing is resident" || bad "something is still resident after unload"

# ------------------------------------- 10. the measurement, and what to paste

if [ "$REMOTE" = "0" ]; then
  get accelerator >"$WORK/accel-after.json"
  AFTER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["used_bytes"])' "$WORK/accel-after.json")"
  DELTA=$(( AFTER - BEFORE ))
  if [ "$DELTA" -gt $((1 << 30)) ]; then
    bad "the card did not come back: $BEFORE -> $AFTER bytes"
  else
    ok "the card came back to where it started ($BEFORE -> $AFTER bytes)"
  fi

  if [ -n "${PEAK_MIB:-}" ] && [ "$PEAK_MIB" -gt 0 ]; then
    BEFORE_MIB=$(( BEFORE / 1048576 ))
    ENGINE_MIB=$(( PEAK_MIB - BEFORE_MIB ))
    ENGINE_BYTES=$(( ENGINE_MIB * 1048576 ))
    printf '\n'
    printf 'MEASURED. Paste into voices/%s.toml, in the [voice.backends.<kind>] block:\n' "$VOICE"
    printf '\n'
    printf '  memory_bytes_estimate = %s\n' "$ENGINE_BYTES"
    printf '  estimate_basis = "measured"\n'
    printf '\n'
    printf '  # MEASURED by scripts/keeper-tts-live.sh on %s.\n' "$(date -u +%Y-%m-%d)"
    printf '  #   card before the engine started  %s MiB\n' "$BEFORE_MIB"
    printf '  #   card PEAK while rendering       %s MiB\n' "$PEAK_MIB"
    printf '  #   => the engine'"'"'s share          %s MiB = %s bytes\n' "$ENGINE_MIB" "$ENGINE_BYTES"
    printf '  # The peak is the figure: it is what the machine must be able to give\n'
    printf '  # while a chunk is in flight, with the desktop excluded. The number it\n'
    printf '  # replaces was DECLARED (%s bytes, from SGLang'"'"'s configured\n' "$DECLARED"
    printf '  # --mem-fraction-static) and never watched on a card.\n'
    printf '\n'
  else
    log "no nvidia-smi samples, so no memory measurement was taken"
  fi
else
  log "remote mode: the card cannot be watched from here, so no memory measurement was taken"
fi

printf '\n%s passed, %s failed\n' "$PASSED" "$FAILED"
[ "$FAILED" -eq 0 ]
