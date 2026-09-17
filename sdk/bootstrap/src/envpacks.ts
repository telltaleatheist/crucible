/**
 * `envpacks.json` — the release's own list of what environments it published.
 *
 * PHASE14-ENVPACKS.md section 2 owns the schema and the asset names; this file
 * is the reader, and it is deliberately strict. A manifest that is missing a
 * field, names zero parts, or carries a digest that is not 64 hex characters is
 * `pack_manifest_unreadable`, not a manifest with holes in it: the whole point
 * of the file is that a machine can be installed from it without asking a
 * person anything, and a half-read manifest is how that turns into a
 * half-installed machine.
 *
 * A `(name, backend)` that is absent is `pack_not_published` — never a quiet
 * build (section 2), never a guess at a neighbouring version.
 */
import { BootstrapRefusal } from './errors.js';

/**
 * The backends a pack is built for.
 *
 * `llama-windows` is the Windows one (PHASE15-HOST.md section 0's amendment
 * and 3.5): a Crucible running natively on Windows with llama-server as its
 * engine. Its one pack, `host`, carries the tray AND that server.
 * `backendFor()` never returns it, because on win32 BOOTSTRAP is talking about
 * the WSL guest, which is Linux — the Windows pack is `install.ps1`'s and the
 * host's, never a guest install's.
 */
export type PackBackend = 'cuda-linux' | 'mlx-darwin' | 'llama-windows';

/** The repository every asset comes from. One owner for the URL shape. */
export const RELEASE_REPO = 'telltaleatheist/crucible';

/** The manifest asset's name, verbatim. */
export const ENVPACKS_ASSET = 'envpacks.json';

/** The server's own pack: the name in the manifest's `packs[].name`. */
export const SERVER_PACK = 'server';

/** The Windows host's pack (PHASE15-HOST.md 4.4), and the backend it is for. */
export const HOST_PACK = 'host';
export const HOST_BACKEND: PackBackend = 'llama-windows';

/** One pack, as the manifest describes it. */
export interface PackEntry {
  name: string;
  backend: PackBackend;
  /** `3.11.13` — the python-build-standalone the pack was built on. */
  python: string;
  /** The REASSEMBLED archive's size. Each part is fetched, appended, deleted. */
  bytes: number;
  /** The REASSEMBLED archive's sha256, lowercase hex. */
  sha256: string;
  /** The part asset names, in the order they are concatenated. Never empty. */
  parts: string[];
  /** sha256 of `envs/<name>/<backend>.txt` at build time. Absent for the server pack. */
  recipeSha256: string | null;
  /** What the unpacked tree costs on disk — what the pre-flight checks. */
  unpackedBytes: number;
  /**
   * The release whose assets hold these bytes, which is not always the release
   * whose manifest carries the row.
   *
   * A pack is a function of its RECIPE, so when a recipe has not changed the
   * pack the previous release built IS the pack this one would build. Rebuilding
   * it wastes CI minutes and copying it forward would store the same 17 GB under
   * every tag, so an unchanged pack is carried by reference and the row keeps
   * naming the release that already has it.
   *
   * Schema 1 had no such field because it did not need one: every pack it named
   * was an asset of its own release. Reading those rows as the manifest's own
   * version is therefore exact, not a default.
   */
  release: string;
}

export interface EnvPacks {
  schema: 1 | 2;
  version: string;
  packs: PackEntry[];
  /** Where this manifest was read from, for every message about it. */
  url: string;
}

/** `https://github.com/<repo>/releases/download/v<release>/<asset>`. */
export function releaseAssetUrl(release: string, asset: string): string {
  if (!/^\d+\.\d+\.\d+/.test(release)) {
    throw new BootstrapRefusal(
      'pack_manifest_unreadable',
      `${JSON.stringify(release)} is not a Crucible version, so there is no release to read packs from.`,
    );
  }
  return `https://github.com/${RELEASE_REPO}/releases/download/v${release}/${asset}`;
}

export function envpacksUrl(release: string): string {
  return releaseAssetUrl(release, ENVPACKS_ASSET);
}

/** `crucible-env-<name>-<backend>-<version>.tar.zst`, exactly (section 1). */
export function packAssetName(name: string, backend: PackBackend, version: string): string {
  return `crucible-env-${name}-${backend}-${version}.tar.zst`;
}

/** `crucible-rootfs-<version>.tar.zst`, the WSL image (section 4b). */
export function rootfsAssetName(version: string): string {
  return `crucible-rootfs-${version}.tar.zst`;
}

/** The backend a platform (or a WSL guest) is. There is no third answer. */
export function backendFor(platform: NodeJS.Platform): PackBackend {
  if (platform === 'darwin') return 'mlx-darwin';
  // win32 means "inside the WSL2 guest", which is Linux x86_64.
  if (platform === 'win32' || platform === 'linux') return 'cuda-linux';
  throw new BootstrapRefusal(
    'unsupported_platform',
    `there is no Crucible backend for ${platform}: cuda-linux (Linux, or WSL2 on Windows) and mlx-darwin are the two`,
  );
}

function bad(url: string, why: string): BootstrapRefusal {
  return new BootstrapRefusal(
    'pack_manifest_unreadable',
    `${url} is not an envpacks manifest this reader accepts: ${why}. `
      + 'A pack is installed from this file and nothing else, so a field that cannot be read is refused rather than skipped.',
  );
}

function requireInt(raw: Record<string, unknown>, key: string, url: string, where: string): number {
  const value = raw[key];
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw bad(url, `${where}.${key} must be a positive integer, got ${JSON.stringify(value)}`);
  }
  return value;
}

function requireString(raw: Record<string, unknown>, key: string, url: string, where: string): string {
  const value = raw[key];
  if (typeof value !== 'string' || value.trim() === '') {
    throw bad(url, `${where}.${key} must be a non-empty string, got ${JSON.stringify(value)}`);
  }
  return value;
}

/** Parse and validate. `release` is what the caller asked for; a manifest for another version is refused. */
export function parseEnvpacks(text: string, url: string, release: string): EnvPacks {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    const head = text.trim().slice(0, 120);
    throw bad(url, `it is not JSON (${(err as Error).message}); it begins ${JSON.stringify(head)}`);
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) throw bad(url, 'the top level is not an object');
  const table = parsed as Record<string, unknown>;
  const schema = table['schema'];
  if (schema !== 1 && schema !== 2) {
    throw bad(url, `schema must be 1 or 2, got ${JSON.stringify(schema)}`);
  }
  const version = requireString(table, 'version', url, 'the manifest');
  if (version !== release) {
    throw bad(url, `it says version ${JSON.stringify(version)} and this is the ${release} release's manifest URL`);
  }
  const rawPacks = table['packs'];
  if (!Array.isArray(rawPacks)) throw bad(url, 'packs must be an array');
  const packs: PackEntry[] = rawPacks.map((entry, index) => {
    const where = `packs[${index}]`;
    if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) throw bad(url, `${where} is not an object`);
    const raw = entry as Record<string, unknown>;
    const name = requireString(raw, 'name', url, where);
    const backend = requireString(raw, 'backend', url, where);
    if (backend !== 'cuda-linux' && backend !== 'mlx-darwin') {
      throw bad(url, `${where}.backend is ${JSON.stringify(backend)}; the two backends are cuda-linux and mlx-darwin`);
    }
    const sha256 = requireString(raw, 'sha256', url, where);
    if (!/^[0-9a-f]{64}$/.test(sha256)) throw bad(url, `${where}.sha256 is not 64 lowercase hex characters`);
    const rawParts = raw['parts'];
    if (!Array.isArray(rawParts) || rawParts.length === 0) throw bad(url, `${where}.parts must list at least one asset`);
    const parts = rawParts.map((part, partIndex) => {
      if (typeof part !== 'string' || part.trim() === '' || part.includes('/')) {
        throw bad(url, `${where}.parts[${partIndex}] must be an asset name, got ${JSON.stringify(part)}`);
      }
      return part;
    });
    const recipe = raw['recipe_sha256'];
    if (recipe !== undefined && recipe !== null && typeof recipe !== 'string') {
      throw bad(url, `${where}.recipe_sha256 must be a string or absent`);
    }
    return {
      name,
      backend,
      python: requireString(raw, 'python', url, where),
      bytes: requireInt(raw, 'bytes', url, where),
      sha256,
      parts,
      recipeSha256: typeof recipe === 'string' ? recipe : null,
      unpackedBytes: requireInt(raw, 'unpacked_bytes', url, where),
      // Required from schema 2. On a schema-1 row this is not a default: that
      // schema placed every pack on its own release by construction, so the
      // manifest's version IS the row's release, stated structurally.
      release: schema >= 2 ? requireString(raw, 'release', url, where) : version,
    };
  });
  return { schema, version, packs, url };
}

/** The pack for a (name, backend), or `pack_not_published` naming what the manifest does have. */
export function findPack(manifest: EnvPacks, name: string, backend: PackBackend): PackEntry {
  const found = manifest.packs.find((pack) => pack.name === name && pack.backend === backend);
  if (found !== undefined) return found;
  const forBackend = manifest.packs.filter((pack) => pack.backend === backend).map((pack) => pack.name);
  throw new BootstrapRefusal(
    'pack_not_published',
    `the ${manifest.version} release publishes no "${name}" pack for ${backend}. `
      + `${manifest.url} lists ${forBackend.length === 0 ? 'nothing' : forBackend.join(', ')} for that backend. `
      + 'A missing pack is never built here; the release is what is installed.',
  );
}
