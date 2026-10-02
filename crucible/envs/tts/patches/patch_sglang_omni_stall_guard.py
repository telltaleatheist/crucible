"""The Higgs stall guard, applied to sglang-omni 0.1.4's Higgs TTS sampler.

A Higgs stall is codebook 0 emitting the same silence code frame after frame and
never leaving: at top-k 50 / top-p 0.95 the exit tokens are cut to zero, so the
state is absorbing. The guard lowers the logit of every cb0 code in a short ring of
recent codes once a row has repeated itself for long enough, before temperature,
top-k and top-p, and lets the model's own next choice take over. Nothing is forced.

THE CONTRACT (identical on narrator's MLX arm, in BookForge):

  HIGGS_STALL_GUARD = "off" | "<frames>,<rate>,<max>,<window>"   e.g. "37,0.5,20,8"
    read once, when sampler.py is imported (the server's startup). Unset = off.
    Anything else that is not exactly that grammar (surrounding whitespace
    stripped; integers as [0-9]+, rate/max as [0-9]+(.[0-9]+)?; frames 1..10000,
    rate 0.001..100, max 0.001..1000, window 1..64) is a ValueError at startup
    naming the variable - including the empty string.

  Per row: a ring R of the last <window> sampled cb0 codes (-1 = an empty slot)
  and a counter run. A row is COUNTED on a frame when it is active (not done), past
  the delay window, not in EOC wind-down, and not finishing on this frame.
    before sampling, for a counted row with run > frames:
      pen = min(max, rate * (run - frames)) is subtracted from the cb0 logit of
      every distinct code in R (once per code, however often R holds it);
    after sampling cb0 code c, for a counted row:
      run = run + 1 if c is in R else 0; then c is pushed into R, the oldest out;
    a row that is NOT counted on a frame has run = 0 and R emptied (the
    prototype's rule: run resets on every frame a row is not counted).
  reset_row() empties both for the row's next owner.

Usage: <env python> patch_sglang_omni_stall_guard.py <env prefix>

Idempotent by marker; all-or-nothing across the three files; pinned to
sglang-omni 0.1.4. A file carrying an OLDER version of this patch is re-derived
from its .orig snapshot.
"""

import glob
import os
import re
import shutil
import sys

PACKAGE_REL = "sglang_omni/models/higgs_tts"
VERSION_REL = "sglang_omni/__init__.py"
EXPECTED_VERSION = "0.1.4"

VERSION = 1
TAG_FAMILY = "# PATCH (crucible stall-guard "
TAG = TAG_FAMILY + f"v{VERSION}, envs/tts/patches/patch_sglang_omni_stall_guard.py)"

SAMPLER_CONFIG = (
    "# CG-baked top-k upper bound = full codec vocab, so the default value is a no-op filter.\n"
    "K_MAX = 1026\n"
    "\n"
    "\n"
    + TAG + ":\n"
    "# the Higgs stall guard. Read once, here, at import; see the applier for the\n"
    "# contract, which narrator's MLX arm implements identically.\n"
    "import os as _stall_os  # noqa: E402\n"
    "import re as _stall_re  # noqa: E402\n"
    "\n"
    "STALL_GUARD_VARIABLE = \"HIGGS_STALL_GUARD\"\n"
    f"STALL_GUARD_PATCH_VERSION = {VERSION}\n"
    "STALL_RING_EMPTY = -1\n"
    "_STALL_INTEGER = _stall_re.compile(r\"[0-9]+\")\n"
    "_STALL_DECIMAL = _stall_re.compile(r\"[0-9]+(\\.[0-9]+)?\")\n"
    "_STALL_RANGES = {\n"
    "    \"frames\": (1, 10000),\n"
    "    \"rate\": (0.001, 100.0),\n"
    "    \"max\": (0.001, 1000.0),\n"
    "    \"window\": (1, 64),\n"
    "}\n"
    "\n"
    "\n"
    "@dataclass(frozen=True)\n"
    "class StallGuard:\n"
    "    frames: int\n"
    "    rate: float\n"
    "    max: float\n"
    "    window: int\n"
    "\n"
    "\n"
    "def parse_stall_guard(raw: str | None) -> \"StallGuard | None\":\n"
    "    if raw is None:\n"
    "        return None\n"
    "    value = raw.strip()\n"
    "    if value == \"off\":\n"
    "        return None\n"
    "    where = f\"{STALL_GUARD_VARIABLE}={raw!r}\"\n"
    "    parts = value.split(\",\")\n"
    "    if len(parts) != 4:\n"
    "        raise ValueError(\n"
    "            f\"{where}: must be 'off' or '<frames>,<rate>,<max>,<window>' \"\n"
    "            \"(e.g. '37,0.5,20,8')\"\n"
    "        )\n"
    "    shapes = (_STALL_INTEGER, _STALL_DECIMAL, _STALL_DECIMAL, _STALL_INTEGER)\n"
    "    names = (\"frames\", \"rate\", \"max\", \"window\")\n"
    "    for name, text, shape in zip(names, parts, shapes):\n"
    "        if not shape.fullmatch(text):\n"
    "            raise ValueError(f\"{where}: {name} {text!r} is not a plain decimal number\")\n"
    "    guard = StallGuard(\n"
    "        frames=int(parts[0]), rate=float(parts[1]), max=float(parts[2]),\n"
    "        window=int(parts[3]),\n"
    "    )\n"
    "    for name in names:\n"
    "        low, high = _STALL_RANGES[name]\n"
    "        if not low <= getattr(guard, name) <= high:\n"
    "            raise ValueError(\n"
    "                f\"{where}: {name} must be in [{low:g}, {high:g}], \"\n"
    "                f\"got {getattr(guard, name):g}\"\n"
    "            )\n"
    "    return guard\n"
    "\n"
    "\n"
    "STALL_GUARD = parse_stall_guard(_stall_os.environ.get(STALL_GUARD_VARIABLE))\n"
    "# The ring's width is fixed for the life of the process, so every buffer that\n"
    "# holds it has one shape and a captured CUDA graph can index it.\n"
    "STALL_RING_WIDTH = 1 if STALL_GUARD is None else STALL_GUARD.window\n"
)

SAMPLER_EDITS = (
    (
        "# CG-baked top-k upper bound = full codec vocab, so the default value is a no-op filter.\n"
        "K_MAX = 1026\n",
        SAMPLER_CONFIG,
    ),
    (
        "        self.step_count = torch.zeros(\n"
        "            self.max_batch_size, dtype=torch.long, device=self.device\n"
        "        )\n"
        "\n"
        "    def reset_row",
        "        self.step_count = torch.zeros(\n"
        "            self.max_batch_size, dtype=torch.long, device=self.device\n"
        "        )\n"
        "        " + TAG + ":\n"
        "        # the stall guard's run counter and the ring of the last\n"
        "        # STALL_RING_WIDTH counted cb0 codes (STALL_RING_EMPTY = an empty slot;\n"
        "        # the filled slots are always the ring's newest, rightmost ones).\n"
        "        self.stall_run = torch.zeros(\n"
        "            self.max_batch_size, dtype=torch.long, device=self.device\n"
        "        )\n"
        "        self.stall_ring = torch.full(\n"
        "            (self.max_batch_size, STALL_RING_WIDTH),\n"
        "            STALL_RING_EMPTY,\n"
        "            dtype=torch.long,\n"
        "            device=self.device,\n"
        "        )\n"
        "\n"
        "    def reset_row",
    ),
    (
        "        self.step_count[row] = 0\n",
        "        self.step_count[row] = 0\n"
        "        self.stall_run[row] = 0\n"
        "        self.stall_ring[row].fill_(STALL_RING_EMPTY)\n",
    ),
    (
        "    seeds = state.seeds[row_indices]\n"
        "    step_count = state.step_count[row_indices]\n",
        "    seeds = state.seeds[row_indices]\n"
        "    step_count = state.step_count[row_indices]\n"
        "    " + TAG + ":\n"
        "    # gathered, updated in place by batched_step_direct and scattered back,\n"
        "    # exactly as the graph path does it.\n"
        "    stall_run = state.stall_run[row_indices] if STALL_GUARD is not None else None\n"
        "    stall_ring = state.stall_ring[row_indices] if STALL_GUARD is not None else None\n",
    ),
    (
        "        seeds=seeds,\n"
        "        step_count=step_count,\n"
        "        boc_id=boc_id,\n"
        "        eoc_id=eoc_id,\n"
        "    )\n"
        "\n"
        "    state.delay_count[row_indices]",
        "        seeds=seeds,\n"
        "        step_count=step_count,\n"
        "        boc_id=boc_id,\n"
        "        eoc_id=eoc_id,\n"
        "        stall_run=stall_run,\n"
        "        stall_ring=stall_ring,\n"
        "    )\n"
        "\n"
        "    if STALL_GUARD is not None:\n"
        "        state.stall_run[row_indices] = stall_run\n"
        "        state.stall_ring[row_indices] = stall_ring\n"
        "    state.delay_count[row_indices]",
    ),
    (
        "    top_p: torch.Tensor | None = None,\n"
        "    top_k_buf: torch.Tensor | None = None,\n"
        "    boc_id: int = BOC_ID,\n"
        "    eoc_id: int = EOC_ID,\n"
        ") -> tuple[",
        "    top_p: torch.Tensor | None = None,\n"
        "    top_k_buf: torch.Tensor | None = None,\n"
        "    boc_id: int = BOC_ID,\n"
        "    eoc_id: int = EOC_ID,\n"
        "    stall_run: torch.Tensor | None = None,\n"
        "    stall_ring: torch.Tensor | None = None,\n"
        ") -> tuple[",
    ),
    (
        "    delay_count = delay_count.to(torch.long)\n"
        "    eoc_countdown = eoc_countdown.to(torch.long)\n"
        "\n"
        "    codes_BN = _sample_independent_batched(\n",
        "    delay_count = delay_count.to(torch.long)\n"
        "    eoc_countdown = eoc_countdown.to(torch.long)\n"
        "\n"
        "    " + TAG + ":\n"
        "    # the penalty, from the state previous frames left. BEFORE temperature,\n"
        "    # top-k and top-p, so a greedy row is covered too. cb0 only, in place:\n"
        "    # gather the ring's codes' logits, subtract, scatter back. A code the ring\n"
        "    # holds twice is written twice with the SAME value, so it is penalised\n"
        "    # once and the write is deterministic; an empty slot is pointed at the\n"
        "    # ring's newest code (always filled when any slot is) for the same reason.\n"
        "    # Fixed shapes and no host branch on a tensor value: CUDA-graph safe.\n"
        "    stall_guard_on = (\n"
        "        STALL_GUARD is not None and stall_run is not None and stall_ring is not None\n"
        "    )\n"
        "    if stall_guard_on:\n"
        "        counted_B = (~generation_done) & (delay_count >= N) & (eoc_countdown < 0)\n"
        "        filled_BW = stall_ring >= 0\n"
        "        over_B = (stall_run - STALL_GUARD.frames).clamp(min=0).to(logits_BNV.dtype)\n"
        "        pen_B = (over_B * STALL_GUARD.rate).clamp(max=STALL_GUARD.max)\n"
        "        pen_B = torch.where(\n"
        "            counted_B & filled_BW.any(dim=1), pen_B, torch.zeros_like(pen_B)\n"
        "        )\n"
        "        slots_BW = torch.where(\n"
        "            filled_BW, stall_ring, stall_ring[:, -1:].clamp(min=0)\n"
        "        )\n"
        "        cb0_logits_BV = logits_BNV[:, 0, :]\n"
        "        cb0_logits_BV.scatter_(\n"
        "            1, slots_BW, cb0_logits_BV.gather(1, slots_BW) - pen_B.view(B, 1)\n"
        "        )\n"
        "\n"
        "    codes_BN = _sample_independent_batched(\n",
    ),
    (
        "    new_step_count = step_count + active.to(step_count.dtype)\n",
        "    new_step_count = step_count + active.to(step_count.dtype)\n"
        "\n"
        "    " + TAG + ":\n"
        "    # the count, after sampling. A row not counted on this frame (done, in the\n"
        "    # delay window, winding down, or finishing now) has its run zeroed and its\n"
        "    # ring emptied; a counted row's run grows when its cb0 code is already in\n"
        "    # the ring and is zeroed when it is not, and the code is then pushed.\n"
        "    if stall_guard_on:\n"
        "        steady_B = (\n"
        "            active & (~in_delay_active) & (~in_winddown_active) & (~done_this_step)\n"
        "        )\n"
        "        cb0_B1 = codes_BN[:, :1].to(stall_ring.dtype)\n"
        "        repeat_B = (stall_ring == cb0_B1).any(dim=1)\n"
        "        pushed_BW = torch.cat([stall_ring[:, 1:], cb0_B1], dim=1)\n"
        "        stall_run.copy_(\n"
        "            torch.where(steady_B & repeat_B, stall_run + 1, torch.zeros_like(stall_run))\n"
        "        )\n"
        "        stall_ring.copy_(\n"
        "            torch.where(\n"
        "                steady_B.unsqueeze(-1),\n"
        "                pushed_BW,\n"
        "                torch.full_like(stall_ring, STALL_RING_EMPTY),\n"
        "            )\n"
        "        )\n",
    ),
)

MODEL_EDITS = (
    (
        "    K_MAX,\n"
        "    NO_SEED,\n",
        "    K_MAX,\n"
        "    NO_SEED,\n"
        "    STALL_RING_EMPTY,\n"
        "    STALL_RING_WIDTH,\n",
    ),
    (
        "        self._cg_active_step_count = torch.zeros(\n"
        "            pool_size, dtype=torch.long, device=cg_device\n"
        "        )\n",
        "        self._cg_active_step_count = torch.zeros(\n"
        "            pool_size, dtype=torch.long, device=cg_device\n"
        "        )\n"
        "        " + TAG + ":\n"
        "        # the stall guard's shadow state, gathered and scattered by the runner\n"
        "        # beside step_count; the captured decode only ever slices it.\n"
        "        self._cg_active_stall_run = torch.zeros(\n"
        "            pool_size, dtype=torch.long, device=cg_device\n"
        "        )\n"
        "        self._cg_active_stall_ring = torch.full(\n"
        "            (pool_size, STALL_RING_WIDTH),\n"
        "            STALL_RING_EMPTY,\n"
        "            dtype=torch.long,\n"
        "            device=cg_device,\n"
        "        )\n",
    ),
    (
        "            seeds=seeds_B,\n"
        "            step_count=step_count_B,\n"
        "        )\n",
        "            seeds=seeds_B,\n"
        "            step_count=step_count_B,\n"
        "            stall_run=self._cg_active_stall_run[:batch_size],\n"
        "            stall_ring=self._cg_active_stall_ring[:batch_size],\n"
        "        )\n",
    ),
)

MODEL_RUNNER_EDITS = (
    (
        "from sglang_omni.models.higgs_tts.sampler import K_MAX, selected_token_logprobs\n",
        "from sglang_omni.models.higgs_tts.sampler import K_MAX, selected_token_logprobs\n"
        + TAG + ":\n"
        "from sglang_omni.models.higgs_tts.sampler import STALL_GUARD\n",
    ),
    (
        "        model._cg_active_step_count[:bs] = pool.step_count[rows_t]\n",
        "        model._cg_active_step_count[:bs] = pool.step_count[rows_t]\n"
        "        if STALL_GUARD is not None:\n"
        "            model._cg_active_stall_run[:bs] = pool.stall_run[rows_t]\n"
        "            model._cg_active_stall_ring[:bs] = pool.stall_ring[rows_t]\n",
    ),
    (
        "        pool.step_count[rows_t] = model._cg_active_step_count[:n_real]\n",
        "        pool.step_count[rows_t] = model._cg_active_step_count[:n_real]\n"
        "        if STALL_GUARD is not None:\n"
        "            pool.stall_run[rows_t] = model._cg_active_stall_run[:n_real]\n"
        "            pool.stall_ring[rows_t] = model._cg_active_stall_ring[:n_real]\n",
    ),
)

EDITS = {
    "sampler.py": SAMPLER_EDITS,
    "model.py": MODEL_EDITS,
    "model_runner.py": MODEL_RUNNER_EDITS,
}

MARKERS = {
    "sampler.py": "STALL_GUARD = parse_stall_guard(_stall_os.environ.get(STALL_GUARD_VARIABLE))",
    "model.py": "stall_ring=self._cg_active_stall_ring[:batch_size],",
    "model_runner.py": "pool.stall_ring[rows_t] = model._cg_active_stall_ring[:n_real]",
}


def package_dir(prefix: str) -> str:
    hits = sorted(
        {
            os.path.realpath(p)
            for p in glob.glob(f"{prefix}/lib/python*/site-packages/{PACKAGE_REL}/sampler.py")
        }
    )
    if not hits:
        raise SystemExit(
            f"NOT_FOUND: no {PACKAGE_REL}/sampler.py under {prefix}/lib/python*/site-packages"
        )
    if len(hits) > 1:
        raise SystemExit(
            f"AMBIGUOUS: {len(hits)} distinct site-packages trees under {prefix}: {hits}"
        )
    return os.path.dirname(hits[0])


def read(path: str) -> str:
    with open(path, encoding="utf-8", newline="") as handle:
        return handle.read()


def fail(code: str, message: str) -> None:
    print(f"{code}: {message}", file=sys.stderr)
    sys.exit(2)


def check_version(package: str) -> None:
    version_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(package))), VERSION_REL
    )
    found = re.search(r'__version__\s*=\s*"([^"]+)"', read(version_path))
    version = found.group(1) if found else None
    if version != EXPECTED_VERSION:
        fail(
            "VERSION_MISMATCH",
            f"this patch was derived against sglang-omni {EXPECTED_VERSION} and "
            f"{version_path} says {version!r}; re-derive it",
        )


def main() -> None:
    prefix = (sys.argv[1] if len(sys.argv) > 1 else "").rstrip("/")
    if not prefix:
        raise SystemExit("usage: patch_sglang_omni_stall_guard.py <env-prefix>")
    package = package_dir(prefix)
    paths = {name: os.path.join(package, name) for name in EDITS}
    live = {name: read(path) for name, path in paths.items()}

    todo = [name for name in EDITS if TAG not in live[name]]
    if not todo:
        for name in EDITS:
            print("ALREADY_PATCHED " + paths[name])
        return

    check_version(package)

    plan: dict[str, tuple[str, bool]] = {}
    for name in todo:
        base, from_orig = live[name], False
        if TAG_FAMILY in base:
            snapshot = paths[name] + ".orig"
            if not os.path.isfile(snapshot):
                fail(
                    "NO_SNAPSHOT",
                    f"{paths[name]} carries an older stall-guard patch and there is "
                    f"no {snapshot} to re-derive it from; reinstall sglang-omni",
                )
            base, from_orig = read(snapshot), True
            if TAG_FAMILY in base:
                fail("SNAPSHOT_PATCHED", f"{snapshot} is not the stock file")
        patched = base
        for old, new in EDITS[name]:
            if patched.count(old) != 1:
                fail(
                    "ANCHOR_NOT_FOUND",
                    f"expected exactly one {old.strip()[:80]!r} in {paths[name]}"
                    + (" (.orig)" if from_orig else "")
                    + f", found {patched.count(old)}",
                )
            patched = patched.replace(old, new)
        if TAG not in patched or MARKERS[name] not in patched:
            fail("INCOMPLETE", f"the stall guard did not land whole in {paths[name]}")
        plan[name] = (patched, from_orig)

    for name, (patched, from_orig) in plan.items():
        if not from_orig:
            shutil.copy2(paths[name], paths[name] + ".orig")
        with open(paths[name], "w", encoding="utf-8", newline="") as handle:
            handle.write(patched)
        print("PATCHED " + paths[name])
    for name in EDITS:
        if name not in plan:
            print("ALREADY_PATCHED " + paths[name])


if __name__ == "__main__":
    main()
