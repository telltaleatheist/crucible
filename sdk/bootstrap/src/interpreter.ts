import { BootstrapRefusal } from './errors.js';
import type { ServerBackend } from './release.js';

export interface StandalonePython {
  /** The CPython version, which also names its stamped directory. */
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
  'llama-windows': {
    version: '3.11.16',
    release: '20260901',
    asset: 'cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only.tar.gz',
    sha256: '6be524fa6752af802146a4adc7d098565425b0b1c166e19a5a7a4c8cccb86bf6',
  },
};

/** The publisher, spelled once. */
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

/** The tray's packages, installed after the wheel on platforms with a desktop. */
export const DESKTOP_PACKAGES: readonly string[] = ['pystray', 'pillow'];
