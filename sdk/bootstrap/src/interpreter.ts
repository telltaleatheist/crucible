/**
 * The CPython an install puts at `<CRUCIBLE_HOME>/server/`, pinned by digest.
 *
 * PHASE20-CODE-NOT-ENVIRONMENTS.md section 2: an interpreter is not code we
 * wrote, so it is not on our releases. It comes from
 * astral-sh/python-build-standalone at a pinned release, verified against a
 * digest read out of that release's own `SHA256SUMS`, and is downloaded ONCE —
 * an upgrade that finds the same digest stamped on disk fetches nothing.
 *
 * `install_only` rather than the full build: one relocatable `python/`
 * directory, no absolute paths baked in, which is the whole reason a tree
 * downloaded into `<CRUCIBLE_HOME>` runs at all. On Windows the same archive
 * unpacks to `python/python.exe`, `python/Scripts/`, `python/Lib/` and no
 * `bin/` whatsoever, which is why nothing here spells either layout inline.
 *
 * TWO TABLES, ONE FACT, TIED BY A CHECK
 * -------------------------------------
 * `crucible/interpreter.py` holds the same pins, because the SERVER downloads a
 * second interpreter when a recipe names a python it does not run. Neither side
 * can import the other — this file runs before any Python exists on the machine,
 * and the generator that writes `install.sh` cannot execute Python — so the two
 * are tied by `tests/test_interpreter.py`, which reads this file as text and
 * asserts every row agrees. It is the seam `crucible/host/wsl_states.py` already
 * uses in the other direction, and a pin edited on one side fails a test rather
 * than silently installing two different interpreters under one version number.
 *
 * This table carries only the 3.11 rows — the server's own. The 3.12 row in the
 * Python table belongs to one RECIPE (`envs/tts/higgs-v3-cuda-linux.txt`) and is
 * downloaded by `crucible install tts`, which happens long after this file's
 * work is done.
 */
import { BootstrapRefusal } from './errors.js';
import type { ServerBackend } from './release.js';

export interface StandalonePython {
  /** `3.11.16` — the CPython version, which is also the directory it is stamped with. */
  version: string;
  /** python-build-standalone's own release tag. */
  release: string;
  /** The asset name on that release. */
  asset: string;
  /** Its sha256, lowercase hex, read from that release's SHA256SUMS. */
  sha256: string;
}

/** The interpreter `pip install <the wheel>` goes into, per backend. */
export const SERVER_PYTHON: Readonly<Record<ServerBackend, StandalonePython>> = {
  'cuda-linux': {
    version: '3.11.16',
    release: '20260901',
    asset: 'cpython-3.11.16+20260901-x86_64-unknown-linux-gnu-install_only.tar.gz',
    sha256: 'faa0758583a63f14c5eee516af82738403b59c13edda6fc0a21d953febd89eed',
  },
  'mlx-darwin': {
    version: '3.11.16',
    release: '20260901',
    asset: 'cpython-3.11.16+20260901-aarch64-apple-darwin-install_only.tar.gz',
    sha256: '50424fa409e8ae84b82a3052522f64695b47dff2158b70bb7358e0ebd6c085c9',
  },
  // The Windows host (PHASE15-HOST.md 4.4). Same release and same CPython as
  // the two backends, which is the property that matters: it runs the same
  // server code. The `-shared` infix PHASE15 first wrote does not exist on
  // 20260901 — the `install_only` build IS the shared one.
  'llama-windows': {
    version: '3.11.16',
    release: '20260901',
    asset: 'cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz',
    sha256: '6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6',
  },
};

/** The publisher, spelled once. Every url below is under it. */
export const STANDALONE_BASE = 'https://github.com/astral-sh/python-build-standalone/releases/download';

export function interpreterUrl(pin: StandalonePython): string {
  return `${STANDALONE_BASE}/${pin.release}/${pin.asset}`;
}

export function interpreterFor(backend: ServerBackend): StandalonePython {
  const pin = SERVER_PYTHON[backend];
  if (pin === undefined) {
    throw new BootstrapRefusal(
      'unsupported_platform',
      `no interpreter is pinned for ${backend}; the backends are ${Object.keys(SERVER_PYTHON).join(', ')}`,
    );
  }
  return pin;
}

/**
 * The two packages the TRAY needs, installed after the wheel on the platforms
 * that have a desktop.
 *
 * NOT DEPENDENCIES OF THE WHEEL, and that is the point: `pyproject.toml` is
 * what every Crucible installs from, so putting them there would make every
 * headless Linux server download a GUI toolkit it can never open a window
 * with. `pystray` draws the icon and menu (PHASE15 4.1/4.2) and `pillow` is
 * what it renders the icon image with.
 */
export const DESKTOP_PACKAGES: readonly string[] = ['pystray', 'pillow'];
