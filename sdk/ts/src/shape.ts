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
