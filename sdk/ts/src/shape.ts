/**
 * The readers for every field this client takes off the wire.
 *
 * ANY CRUCIBLE THAT ANSWERS WORKS (Owen, 2026-09-24): *"lets modify bookforge
 * and foundry so they dont require any particular crucible server. if it can
 * make the call to the crucible server then it should work."* That ruling
 * replaced the lockstep one of 2026-09-20, under which this client demanded
 * every field it knew and a client one release ahead of its server refused it
 * by name. What that cost was measured, not argued: after the 1.0.24 repin,
 * ten BookForge keepers failed only because their fake servers lacked fields
 * that had just been demanded (`capability.classes[0] has no field "work"`,
 * `voices[0] has no field "orphan"`) — and one new informational voice field
 * broke every `info()` probe, because the voice rows ride inside it.
 *
 * So there are TWO kinds of field, and two families of reader:
 *
 * - **Load-bearing** — the fields a call cannot be done right without: an id,
 *   a job's status, the artifacts to fetch, the audio, a chat's content, an
 *   error's code, anything this client's own logic branches on. Read with the
 *   strict readers (`str`, `num`, `bool`, `objectField`, `nullable*`, `oneOf`),
 *   which throw {@link CrucibleProtocolError} naming the field. A missing one is
 *   still refused by name, because nothing true can be said without it.
 * - **Informational** — a field that informs a display, or a number a caller
 *   may use: an estimate, a note, a count, a timestamp, a provenance string, a
 *   measurement. Read with the `opt*` readers below, which answer `null` when
 *   the field is absent (or null) and never throw for absence. Absence is how
 *   an older server looks, and version skew is weather, not misconfiguration:
 *   it is not a reason to refuse a call that worked.
 *
 * **A PRESENT, WRONG-TYPED informational field is still a protocol error.**
 * That is the one line this file draws inside the tolerant family, and it is
 * drawn deliberately. API v1 evolves ADDITIVELY — fields are added, never
 * retyped; a retype is a breaking change that moves `api_version`, and the 426
 * handshake refuses that by name — so absence is the only shape version skew
 * can take. `work: "banana"` is not an old server, it is a broken one, and
 * answering `null` for it would hide the defect behind something that reads
 * like "this server does not say" — the band-aid the no-fallbacks rule has
 * always been about. Where one malformed row would otherwise take a whole
 * probe down with it, the CALLER contains it (`info()` carries an unreadable
 * voice or model row aside rather than failing), which is the owner of that
 * decision; the reader does not soften the fact.
 *
 * Nothing in either family substitutes a value. `null` is what the `opt*`
 * readers say for "not stated", and every type that carries one says so.
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

/**
 * A LOAD-BEARING JSON object that the server may honestly answer `null` to.
 *
 * The key must be present, and that is the whole reason this reader exists
 * beside {@link optObject}: it is for the few fields whose `null` is a
 * statement a caller ACTS on, so that an absent key cannot be mistaken for it.
 * `activity.resident.held_by` is the example — `null` there means "the card is
 * stranded", and a reconciler unloads on exactly that; an older server that
 * does not report the holder must not read as one reporting none.
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

/**
 * A value from a closed vocabulary. For LOAD-BEARING enums only — a job's or a
 * task's state, a decision's type — where this client or its caller branches
 * on the word and a new one would be a contract change. An informational enum
 * (a voice's kind, an estimate's basis, a lane's health) is read with
 * {@link optStr} and carried as the server's own word instead, so a newer
 * server's fourth word is news for a display rather than a lost call.
 */
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

// ------------------------------------------------------------ informational
//
// The `opt*` family. Absent or `null` → `null`; present and the right type →
// the value; present and the wrong type → CrucibleProtocolError. See the
// module doc for why the third is not `null`.

/** The raw value of an informational field, or `null` when it is absent or null. */
function optRaw(object: Json, key: string): unknown {
  const value = object[key];
  return value === undefined ? null : value;
}

function wrongType(where: string, key: string, wanted: string, value: unknown): CrucibleProtocolError {
  return new CrucibleProtocolError(
    `${where}.${key} is present but is not ${wanted} (got ${describe(value)}). ` +
      'An informational field may be absent — an older server reads as null — ' +
      'but one that is sent must be the type API v1 gives it.',
  );
}

export function optStr(object: Json, key: string, where: string): string | null {
  const value = optRaw(object, key);
  if (value === null) return null;
  if (typeof value !== 'string') throw wrongType(where, key, 'a string', value);
  return value;
}

export function optNum(object: Json, key: string, where: string): number | null {
  const value = optRaw(object, key);
  if (value === null) return null;
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw wrongType(where, key, 'a number', value);
  }
  return value;
}

export function optBool(object: Json, key: string, where: string): boolean | null {
  const value = optRaw(object, key);
  if (value === null) return null;
  if (typeof value !== 'boolean') throw wrongType(where, key, 'a boolean', value);
  return value;
}

/** An informational JSON object, returned unopened. */
export function optObject(object: Json, key: string, where: string): Json | null {
  const value = optRaw(object, key);
  if (value === null) return null;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw wrongType(where, key, 'a JSON object', value);
  }
  return value as Json;
}

/** An informational JSON array, returned with its entries unchecked. */
export function optArray(object: Json, key: string, where: string): unknown[] | null {
  const value = optRaw(object, key);
  if (value === null) return null;
  if (!Array.isArray(value)) throw wrongType(where, key, 'a JSON array', value);
  return value;
}

export function optStrArray(object: Json, key: string, where: string): string[] | null {
  const value = optArray(object, key, where);
  if (value === null) return null;
  return value.map((entry, index) => {
    if (typeof entry !== 'string') {
      throw new CrucibleProtocolError(
        `${where}.${key}[${index}] is not a string (got ${describe(entry)})`,
      );
    }
    return entry;
  });
}

function describe(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'an array';
  return typeof value;
}
