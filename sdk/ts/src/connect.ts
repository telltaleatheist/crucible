import type { Pairing } from './pairing.js';
import { API_VERSION } from './types.js';

export const DEFAULT_CRUCIBLE_PORT = 7100;
export interface PairingOptions {
  fetch?: typeof globalThis.fetch;
  signal?: AbortSignal;
  timeoutMs?: number;
}
export interface PairingRequest {
  url: string;
  name: string;
  id: string;
  deviceCode: string;
  userCode: string;
  expiresIn: number;
  interval: number;
  /** Whether an operator must approve this request before it becomes a pairing. */
  approvalRequired: boolean;
}
export interface PendingPairing {
  /** What an approval names. */
  id: string;
  /** The code the person checks against the requesting screen. */
  userCode: string;
  /** Who is asking. */
  clientName: string;
  address: string;
  expiresIn: number;
}
export type PairingResult =
  | { status: 'pending' | 'denied' | 'expired' }
  | { status: 'approved'; pairing: Pairing };

export class CrucibleConnectionError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = 'CrucibleConnectionError';
  }
}

/** Bare addresses use Crucible's standard port; explicit ports are never changed. */
export function crucibleAddress(address: string): string {
  let raw = address.trim();
  if (!raw || /[\s@?#]/.test(raw)) throw new CrucibleConnectionError('invalid_address', 'Enter an IP address, hostname, or server URL without credentials');
  if (!raw.includes('://') && !raw.startsWith('[') && (raw.match(/:/g)?.length ?? 0) > 1) raw = `[${raw}]`;
  const match = /^(?:(https?):\/\/)?(\[[^\]]+\]|[^:/]+)(?::([0-9]+))?\/?$/.exec(raw);
  if (!match) throw new CrucibleConnectionError('invalid_address', 'Enter the server address without a path, query, or token');
  const scheme = match[1] ?? 'http';
  const port = match[3] === undefined ? (scheme === 'https' ? 443 : DEFAULT_CRUCIBLE_PORT) : Number(match[3]);
  if (port < 1 || port > 65535) throw new CrucibleConnectionError('invalid_address', 'The server port must be between 1 and 65535');
  try { return new URL(`${scheme}://${match[2]}:${port}`).origin; }
  catch { throw new CrucibleConnectionError('invalid_address', 'The server address is not valid'); }
}

/**
 * Whether a server URL names a machine on this device's own network: a private IPv4
 * (RFC 1918), a link-local address, an IPv6 unique-local or link-local address, a
 * `.local` name, or a bare computer name. Tailnet (100.64/10) and public addresses are not.
 */
export function looksLikeLanAddress(url: string): boolean {
  const host = new URL(url).hostname.replace(/^\[|\]$/g, '').toLowerCase();
  const v4 = /^(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/.exec(host);
  if (v4) {
    const [a, b] = [Number(v4[1]), Number(v4[2])];
    return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 169 && b === 254);
  }
  if (host.includes(':')) return /^f[cd][0-9a-f]{0,2}:/.test(host) || /^fe[89ab][0-9a-f]?:/.test(host);
  return host.endsWith('.local') || (!host.includes('.') && host !== 'localhost');
}

/**
 * What to say when a machine on this network does not answer. Nothing answering at all is
 * what a loopback-only Crucible looks like from another device: the server never sees the
 * connection, so it cannot refuse it by name, and this is the only place the cause can be
 * named. On Windows the usual one is that sharing was never turned on.
 */
function lanSilence(where: string, how: string): string {
  return `${how} at ${where}. If Crucible runs on that computer, it answers only that computer `
    + 'until it is opened to the network: on Windows, run `crucible lan enable` in PowerShell there '
    + '(or, in the Crucible window, Settings, then Share under "Share on your network"); it also says '
    + 'if Windows has that network marked Public, which keeps other devices out. Then check that this '
    + 'device is on the same network (guest Wi-Fi often keeps devices apart) and, on an iPhone or '
    + 'iPad, that this app is allowed Local Network access (Settings > Privacy & Security > Local Network).';
}

function silence(url: string, timedOut: boolean, timeoutMs: number): CrucibleConnectionError {
  const where = new URL(url).host;
  const how = timedOut ? `Nothing answered within ${Math.round(timeoutMs / 1000)} s` : 'Nothing answered';
  if (looksLikeLanAddress(url)) {
    return new CrucibleConnectionError(timedOut ? 'connection_timed_out' : 'connection_unreachable', lanSilence(where, how));
  }
  return new CrucibleConnectionError(timedOut ? 'connection_timed_out' : 'connection_unreachable',
    `${how} at ${where}. Check the address, and that Crucible is running on that computer and shared with the network it is reached on.`);
}

async function read(url: string, path: string, options: PairingOptions, body?: unknown): Promise<Record<string, unknown>> {
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? 10_000;
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  const abort = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener('abort', abort, { once: true });
  if (options.signal?.aborted) abort();
  const unanswered = (): CrucibleConnectionError => (timedOut || !controller.signal.aborted
    ? silence(url, timedOut, timeoutMs)
    : new CrucibleConnectionError('connection_cancelled', 'The connection check was cancelled'));
  try {
    let response: Response;
    try {
      response = await (options.fetch ?? globalThis.fetch)(url + path, {
        method: body === undefined ? 'GET' : 'POST', redirect: 'error',
        headers: { 'X-Crucible-Api': String(API_VERSION), 'Content-Type': 'application/json' },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: controller.signal,
      });
    } catch {
      throw unanswered();
    }
    let value: unknown;
    try {
      value = await response.json();
    } catch {
      if (controller.signal.aborted) throw unanswered();
      throw new CrucibleConnectionError('not_crucible', `${new URL(url).host} answered HTTP ${response.status}, but not as a Crucible`);
    }
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new CrucibleConnectionError('invalid_response', 'The address did not return a Crucible response');
    const object = value as Record<string, unknown>;
    if (!response.ok) {
      const error = object['error'] as { code?: unknown; message?: unknown } | undefined;
      throw new CrucibleConnectionError(typeof error?.code === 'string' ? error.code : 'connection_refused',
        typeof error?.message === 'string' ? error.message : `Crucible returned HTTP ${response.status}`);
    }
    return object;
  } finally {
    clearTimeout(timeout);
    options.signal?.removeEventListener('abort', abort);
  }
}

export async function startPairing(address: string, clientName: string, options: PairingOptions = {}): Promise<PairingRequest> {
  const url = crucibleAddress(address);
  const ping = await read(url, '/v1/ping', options);
  if (ping['crucible'] !== true || typeof ping['name'] !== 'string' || typeof ping['api_version'] !== 'number') {
    throw new CrucibleConnectionError('not_crucible', 'This address is not a compatible Crucible engine');
  }
  if (ping['api_version'] !== API_VERSION) {
    throw new CrucibleConnectionError(
      'api_version_mismatch',
      `api_version_mismatch: this Crucible speaks API version ${ping['api_version']} and this `
      + `client speaks ${API_VERSION}. Update whichever of the two is older.`,
    );
  }
  if (ping['pairing_version'] !== 1) throw new CrucibleConnectionError('pairing_unavailable', 'Update Crucible on that computer to use approval pairing, or paste its existing connection line');
  const value = await read(url, '/v1/pairing/start', options, { client_name: clientName });
  if (value['name'] !== ping['name'] || typeof value['id'] !== 'string' || typeof value['device_code'] !== 'string'
    || typeof value['user_code'] !== 'string' || typeof value['expires_in'] !== 'number' || typeof value['interval'] !== 'number'
    || typeof value['approval_required'] !== 'boolean' || value['expires_in'] <= 0 || value['interval'] < 1) {
    throw new CrucibleConnectionError('invalid_response', 'Crucible returned an incompatible pairing request');
  }
  return { url, name: value['name'] as string, id: value['id'], deviceCode: value['device_code'],
    userCode: value['user_code'], expiresIn: value['expires_in'], interval: value['interval'],
    approvalRequired: value['approval_required'] };
}

export async function pollPairing(request: PairingRequest, options: PairingOptions = {}): Promise<PairingResult> {
  const value = await read(request.url, '/v1/pairing/poll', options, { id: request.id, device_code: request.deviceCode });
  const status = value['status'];
  if (status === 'pending' || status === 'denied' || status === 'expired') return { status };
  if (status !== 'approved' || value['name'] !== request.name || typeof value['token'] !== 'string' || !value['token']) {
    throw new CrucibleConnectionError('invalid_response', 'Crucible returned an incompatible pairing decision');
  }
  return { status, pairing: { name: request.name, url: request.url, token: value['token'] } };
}
