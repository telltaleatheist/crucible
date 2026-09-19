/**
 * What a Crucible release CARRIES, and how to name each asset.
 *
 * PHASE20-CODE-NOT-ENVIRONMENTS.md section 1. **A release carries our code and
 * nothing that is published elsewhere:** the wheel, the sdist, the two SDK
 * tarballs and the two generated installers. Not an interpreter, not an
 * environment, not a WSL image — those come from python-build-standalone, from
 * PyPI and from Canonical, each pinned where the thing that needs it lives.
 *
 * This file was `envpacks.ts`, the reader of a per-release manifest of
 * environment archives. There is no manifest now, because there is nothing for
 * one to describe: an install downloads one interpreter at a digest this
 * package pins and one wheel at a digest the release publishes beside it, and
 * both of those are named here rather than looked up.
 */
import { BootstrapRefusal } from './errors.js';

/**
 * The backend a machine SERVES on.
 *
 * `llama-windows` is the Windows one (PHASE15-HOST.md section 0's amendment
 * and 3.5): a Crucible running natively on Windows with llama-server as its
 * engine. `backendFor()` never returns it, because on win32 BOOTSTRAP is
 * talking about the WSL guest, which is Linux — the Windows runtime is
 * `install.ps1`'s and the host's, never a guest install's.
 */
export type ServerBackend = 'cuda-linux' | 'mlx-darwin' | 'llama-windows';

/** The repository every asset comes from. One owner for the URL shape. */
export const RELEASE_REPO = 'telltaleatheist/crucible';

/** The backend the Windows host runs (PHASE15-HOST.md 4.4). */
export const HOST_BACKEND: ServerBackend = 'llama-windows';

/** `https://github.com/<repo>/releases/download/v<release>/<asset>`. */
export function releaseAssetUrl(release: string, asset: string): string {
  if (!/^\d+\.\d+\.\d+/.test(release)) {
    // The same name an unreadable channel answer gets, and for the same
    // reason `errors.ts` gives there: a version nothing can order is a version
    // nothing can install, whether it came from a channel or from a caller.
    throw new BootstrapRefusal(
      'release_channel_unreadable',
      `${JSON.stringify(release)} is not a Crucible version, so there is no release to install from.`,
    );
  }
  return `https://github.com/${RELEASE_REPO}/releases/download/v${release}/${asset}`;
}

/**
 * `crucible-<version>-py3-none-any.whl` — THE deploy.
 *
 * Pure python and `py3-none-any`, so one file installs on every backend: what
 * differs between a Linux server and a Mac one is the interpreter under it and
 * the recipes it later pips, never this.
 */
export function wheelAssetName(release: string): string {
  return `crucible-${release}-py3-none-any.whl`;
}

/**
 * `<wheel>.sha256` — the digest the release publishes beside the wheel, one
 * line with the digest first, the same shape `sha256sum` writes.
 *
 * WHY A SIBLING ASSET AND NOT A FIELD SOMEWHERE. The installer runs before
 * there is any Crucible on the machine to ask, so the only thing it can read
 * is another file on the same release; and a digest that travelled with the
 * bytes it describes would not be a check.
 */
export function wheelShaAssetName(release: string): string {
  return `${wheelAssetName(release)}.sha256`;
}

export function wheelUrl(release: string): string {
  return releaseAssetUrl(release, wheelAssetName(release));
}

export function wheelShaUrl(release: string): string {
  return releaseAssetUrl(release, wheelShaAssetName(release));
}

/** The backend a platform (or a WSL guest) is. There is no third answer. */
export function backendFor(platform: NodeJS.Platform): ServerBackend {
  if (platform === 'darwin') return 'mlx-darwin';
  // win32 means "inside the WSL2 guest", which is Linux x86_64.
  if (platform === 'win32' || platform === 'linux') return 'cuda-linux';
  throw new BootstrapRefusal(
    'unsupported_platform',
    `there is no Crucible backend for ${platform}: cuda-linux (Linux, or WSL2 on Windows) and mlx-darwin are the two`,
  );
}
