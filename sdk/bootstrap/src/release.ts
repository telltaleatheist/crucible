import { BootstrapRefusal } from './errors.js';

/** The backend a machine serves on. */
export type ServerBackend = 'cuda-linux' | 'mlx-darwin' | 'llama-windows';

/** The repository every asset comes from. */
export const RELEASE_REPO = 'telltaleatheist/crucible';

/** The backend the Windows host runs. */
export const HOST_BACKEND: ServerBackend = 'llama-windows';

/** `https://github.com/<repo>/releases/download/v<release>/<asset>`. */
export function releaseAssetUrl(release: string, asset: string): string {
  if (!/^\d+\.\d+\.\d+/.test(release)) {
    throw new BootstrapRefusal(
      'release_channel_unreadable',
      `${JSON.stringify(release)} is not a Crucible version, so there is no release to install from.`,
    );
  }
  return `https://github.com/${RELEASE_REPO}/releases/download/v${release}/${asset}`;
}

/** `crucible-<version>-py3-none-any.whl`, the release's wheel. */
export function wheelAssetName(release: string): string {
  return `crucible-${release}-py3-none-any.whl`;
}

/** `<wheel>.sha256`, the digest the release publishes beside the wheel. */
export function wheelShaAssetName(release: string): string {
  return `${wheelAssetName(release)}.sha256`;
}

export function wheelUrl(release: string): string {
  return releaseAssetUrl(release, wheelAssetName(release));
}

export function wheelShaUrl(release: string): string {
  return releaseAssetUrl(release, wheelShaAssetName(release));
}

/** The backend a platform (or a WSL guest) is. */
export function backendFor(platform: NodeJS.Platform): ServerBackend {
  if (platform === 'darwin') return 'mlx-darwin';
  if (platform === 'win32' || platform === 'linux') return 'cuda-linux';
  throw new BootstrapRefusal(
    'unsupported_platform',
    `there is no Crucible backend for ${platform}: cuda-linux (Linux, or WSL2 on Windows) and mlx-darwin are the two`,
  );
}
