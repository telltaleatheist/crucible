/**
 * Strict readers for the fields API v1 promises.
 *
 * Every one of these throws {@link CrucibleProtocolError} naming the field it
 * could not read. Nothing here substitutes a default: a field the contract says
 * is always present, and which is not present, is a bug on one side of the seam
 * and has to surface as one.
 */

import { CrucibleProtocolError } from './errors.js';

export type Json = Record<string, unknown>;

export function asObject(value: unknown, where: string): Json {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new CrucibleProtocolError(`${where} is not a JSON object (got ${describe(value)})`);
  }
  return value as Json;
}

export function asArray(value: unknown, where: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new CrucibleProtocolError(`${where} is not a JSON array (got ${describe(value)})`);
  }
  return value;
}

export function field(object: Json, key: string, where: string): unknown {
  if (!(key in object)) {
    throw new CrucibleProtocolError(`${where} has no field ${JSON.stringify(key)}`);
  }
  return object[key];
}

export function str(object: Json, key: string, where: string): string {
  const value = field(object, key, where);
  if (typeof value !== 'string') {
    throw new CrucibleProtocolError(`${where}.${key} is not a string (got ${describe(value)})`);
  }
  return value;
}

export function num(object: Json, key: string, where: string): number {
  const value = field(object, key, where);
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new CrucibleProtocolError(`${where}.${key} is not a number (got ${describe(value)})`);
  }
  return value;
}

export function bool(object: Json, key: string, where: string): boolean {
  const value = field(object, key, where);
  if (typeof value !== 'boolean') {
    throw new CrucibleProtocolError(`${where}.${key} is not a boolean (got ${describe(value)})`);
  }
  return value;
}

export function nullableStr(object: Json, key: string, where: string): string | null {
  const value = field(object, key, where);
  if (value === null) return null;
  if (typeof value !== 'string') {
    throw new CrucibleProtocolError(
      `${where}.${key} is neither a string nor null (got ${describe(value)})`,
    );
  }
  return value;
}

/**
 * A boolean that the server may honestly answer `null` to.
 *
 * The key still has to be present. `null` is a statement — on a `chunk` event it
 * is "narrator did not say whether this chunk hit the frame cap" — and a missing
 * key is not that statement, so the two are not collapsed. Nothing here turns
 * either one into `false`: the whole reason `capped` is nullable is that
 * `false` would mean "this was a long sentence, not a runaway".
 */
export function nullableBool(object: Json, key: string, where: string): boolean | null {
  const value = field(object, key, where);
  if (value === null) return null;
  if (typeof value !== 'boolean') {
    throw new CrucibleProtocolError(
      `${where}.${key} is neither a boolean nor null (got ${describe(value)})`,
    );
  }
  return value;
}

export function nullableNum(object: Json, key: string, where: string): number | null {
  const value = field(object, key, where);
  if (value === null) return null;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new CrucibleProtocolError(
      `${where}.${key} is neither a number nor null (got ${describe(value)})`,
    );
  }
  return value;
}

/**
 * A JSON object that the server may honestly answer `null` to, read **without
 * being opened**.
 *
 * The key must be present, for {@link nullableBool}'s reason: `null` is a
 * statement the server made and an absent key is not that statement. What is
 * different here is that nothing inside is checked. This is the reader for a
 * field whose contents belong to somebody else — a `chunk` event's `guard` is
 * the verdict narrator's own retake ladder reached, forwarded verbatim by a
 * server that reads nothing inside it (PHASE6-REMOTE-RENDER.md sections 3 and
 * 4) — and a client that validated the ladder's vocabulary would reject a
 * perfectly good render the day the ladder grew a rung. So: an object, or null,
 * and no opinion about either.
 */
export function nullableObject(object: Json, key: string, where: string): Json | null {
  const value = field(object, key, where);
  if (value === null) return null;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new CrucibleProtocolError(
      `${where}.${key} is neither a JSON object nor null (got ${describe(value)})`,
    );
  }
  return value as Json;
}

export function strArray(object: Json, key: string, where: string): string[] {
  const value = asArray(field(object, key, where), `${where}.${key}`);
  return value.map((entry, index) => {
    if (typeof entry !== 'string') {
      throw new CrucibleProtocolError(
        `${where}.${key}[${index}] is not a string (got ${describe(entry)})`,
      );
    }
    return entry;
  });
}

export function objectField(object: Json, key: string, where: string): Json {
  return asObject(field(object, key, where), `${where}.${key}`);
}

export function oneOf<T extends string>(
  value: string,
  allowed: readonly T[],
  where: string,
): T {
  if (!(allowed as readonly string[]).includes(value)) {
    throw new CrucibleProtocolError(
      `${where} is ${JSON.stringify(value)}, which is not one of ${allowed.join(', ')}`,
    );
  }
  return value as T;
}

function describe(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'an array';
  return typeof value;
}
