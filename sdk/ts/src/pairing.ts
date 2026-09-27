import { CruciblePairingError } from './errors.js';

/** The scheme the operator page prints and a connect door accepts. */
export const PAIRING_SCHEME = 'crucible';

const PREFIX = `${PAIRING_SCHEME}://`;

const HOST_PORT = /^(\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]+):([0-9]{1,5})$/;

/** What a pairing line carries: everything a connect door's three fields need. */
export interface Pairing {
  /** The server's name, percent-decoded. */
  readonly name: string;
  /** The base URL, with no trailing slash and no `/v1` — `CrucibleClient`'s shape. */
  readonly url: string;
  /** The bearer token, percent-decoded. */
  readonly token: string;
}

function elide(line: string): string {
  const hash = line.indexOf('#');
  const shown = hash < 0 ? line : `${line.slice(0, hash)}#…`;
  return shown.length > 120 ? `${shown.slice(0, 117)}...` : shown;
}

function refuse(line: string, detail: string): never {
  throw new CruciblePairingError(detail, elide(line));
}

function decode(raw: string, what: string, line: string): string {
  let value: string;
  try {
    value = decodeURIComponent(raw);
  } catch {
    refuse(line, `the ${what} is not valid percent-encoding`);
  }
  if (value === '') refuse(line, `the ${what} is empty`);
  return value;
}

/** Read one `crucible://` line, or refuse it by name (`invalid_pairing`). */
export function parsePairing(line: string): Pairing {
  if (typeof line !== 'string') {
    throw new CruciblePairingError('a pairing line is a string', '');
  }
  const raw = line.trim();
  if (raw === '') {
    throw new CruciblePairingError('a pairing line is empty', '');
  }
  if (!raw.startsWith(PREFIX)) {
    refuse(raw, `a pairing line starts with ${PREFIX}`);
  }

  const rest = raw.slice(PREFIX.length);
  const hash = rest.indexOf('#');
  if (hash < 0) {
    refuse(raw, 'a pairing line carries its token in a #fragment, and there is none');
  }
  const address = rest.slice(0, hash);
  const fragment = rest.slice(hash + 1);

  if (!address.endsWith('/')) {
    refuse(
      raw,
      'a pairing line ends its address with / before the #; without it the ' +
        'fragment runs into the port',
    );
  }
  const authority = address.slice(0, -1);
  if (authority.includes('/')) {
    refuse(raw, 'a pairing line has no path: it is scheme, name, address and token');
  }

  const at = authority.indexOf('@');
  if (at < 0) {
    refuse(raw, 'a pairing line names its server: crucible://<name>@<host>:<port>/#<token>');
  }
  if (authority.indexOf('@', at + 1) >= 0) {
    refuse(
      raw,
      'there are two @ in the address. A server name contains one, so the name ' +
        'is percent-encoded (%40) — this line was not written by a Crucible, ' +
        'and which @ was meant is not something to guess at',
    );
  }

  const name = decode(authority.slice(0, at), 'server name', raw);
  const hostPort = authority.slice(at + 1);
  const matched = HOST_PORT.exec(hostPort);
  if (matched === null) {
    refuse(
      raw,
      `${JSON.stringify(hostPort)} is not <host>:<port>; a pairing line always ` +
        'states the port',
    );
  }
  const port = Number(matched[2]);
  if (port < 1 || port > 65535) {
    refuse(raw, `port ${port} is not a port`);
  }

  return { name, url: `http://${hostPort}`, token: decode(fragment, 'token', raw) };
}
