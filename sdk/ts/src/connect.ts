/** Discover the canonical endpoint and pair after an operator approves the code. */
import type { Pairing } from './pairing.js';

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
}
export interface PendingPairing {
  id: string;
  userCode: string;
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

async function read(url: string, path: string, options: PairingOptions, body?: unknown): Promise<Record<string, unknown>> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? 10_000);
  const abort = () => controller.abort(options.signal?.reason);
  options.signal?.addEventListener('abort', abort, { once: true });
  if (options.signal?.aborted) abort();
  try {
    const response = await (options.fetch ?? globalThis.fetch)(url + path, {
      method: body === undefined ? 'GET' : 'POST', redirect: 'error',
      headers: { 'X-Crucible-Api': '1', 'Content-Type': 'application/json' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: controller.signal,
    });
    const value: unknown = await response.json();
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new CrucibleConnectionError('invalid_response', 'The address did not return a Crucible response');
    const object = value as Record<string, unknown>;
    if (!response.ok) {
      const error = object['error'] as { code?: unknown; message?: unknown } | undefined;
      throw new CrucibleConnectionError(typeof error?.code === 'string' ? error.code : 'connection_refused',
        typeof error?.message === 'string' ? error.message : `Crucible returned HTTP ${response.status}`);
    }
    return object;
  } catch (error) {
    if (error instanceof CrucibleConnectionError) throw error;
    throw new CrucibleConnectionError(controller.signal.aborted ? 'connection_cancelled' : 'connection_unreachable',
      controller.signal.aborted ? 'The connection check was cancelled or timed out' : 'Crucible did not answer at this address. Check the address and that network sharing is enabled on that computer.');
  } finally {
    clearTimeout(timeout);
    options.signal?.removeEventListener('abort', abort);
  }
}

export async function startPairing(address: string, clientName: string, options: PairingOptions = {}): Promise<PairingRequest> {
  const url = crucibleAddress(address);
  const ping = await read(url, '/v1/ping', options);
  if (ping['crucible'] !== true || typeof ping['name'] !== 'string' || ping['api_version'] !== 1) {
    throw new CrucibleConnectionError('not_crucible', 'This address is not a compatible Crucible engine');
  }
  if (ping['pairing_version'] !== 1) throw new CrucibleConnectionError('pairing_unavailable', 'Update Crucible on that computer to use approval pairing, or paste its existing connection line');
  const value = await read(url, '/v1/pairing/start', options, { client_name: clientName });
  if (value['name'] !== ping['name'] || typeof value['id'] !== 'string' || typeof value['device_code'] !== 'string'
    || typeof value['user_code'] !== 'string' || typeof value['expires_in'] !== 'number' || typeof value['interval'] !== 'number'
    || value['expires_in'] <= 0 || value['interval'] < 1) {
    throw new CrucibleConnectionError('invalid_response', 'Crucible returned an incompatible pairing request');
  }
  return { url, name: value['name'] as string, id: value['id'], deviceCode: value['device_code'],
    userCode: value['user_code'], expiresIn: value['expires_in'], interval: value['interval'] };
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
