/**
 * Base64 for inline job inputs, and for the audio a streaming session sends back.
 *
 * Written out rather than reached for, because the three runtimes this client
 * supports disagree: `Buffer` is Node and Electron only, and `btoa` needs a
 * binary string which `String.fromCharCode(...bytes)` cannot build for a large
 * input without overflowing the call stack. Twenty lines is cheaper than either
 * caveat, and it keeps the runtime-dependency count at zero.
 */

const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

export function encodeBase64(bytes: Uint8Array): string {
  const pieces: string[] = [];
  const limit = bytes.length - (bytes.length % 3);

  for (let index = 0; index < limit; index += 3) {
    const triple =
      (bytes[index] as number) * 65536 +
      (bytes[index + 1] as number) * 256 +
      (bytes[index + 2] as number);
    pieces.push(
      ALPHABET[(triple >> 18) & 63]! +
        ALPHABET[(triple >> 12) & 63]! +
        ALPHABET[(triple >> 6) & 63]! +
        ALPHABET[triple & 63]!,
    );
  }

  const remainder = bytes.length - limit;
  if (remainder === 1) {
    const value = bytes[limit] as number;
    pieces.push(`${ALPHABET[value >> 2]!}${ALPHABET[(value << 4) & 63]!}==`);
  } else if (remainder === 2) {
    const value = (bytes[limit] as number) * 256 + (bytes[limit + 1] as number);
    pieces.push(
      `${ALPHABET[value >> 10]!}${ALPHABET[(value >> 4) & 63]!}${ALPHABET[(value << 2) & 63]!}=`,
    );
  }

  return pieces.join('');
}

/**
 * The other direction, for the PCM a streaming session carries.
 *
 * Written out for the same reason `encodeBase64` is: `Buffer` is Node and
 * Electron only and `atob` is not in every runtime this client supports, and a
 * client with zero runtime dependencies cannot reach for a polyfill.
 *
 * **It refuses malformed input rather than skipping it.** `atob` and most
 * hand-rolled decoders ignore a character outside the alphabet, which turns one
 * corrupted byte on the wire into audio that is quietly a few samples short and
 * shifted for the rest of the chunk. A session's audio is concatenated with its
 * neighbours, so "quietly a few samples short" is a click in the middle of a
 * sentence that nothing in the pipeline would explain.
 */
export function decodeBase64(text: string): Uint8Array {
  const body = text.endsWith('==')
    ? text.slice(0, -2)
    : text.endsWith('=')
      ? text.slice(0, -1)
      : text;
  if (text.length % 4 !== 0 || body.length % 4 === 1) {
    throw new Error(`base64 of length ${text.length} is not a whole number of groups`);
  }

  // Six bits per character, eight per byte: whole bytes only, and the padding
  // is exactly what says how many of the last group's bits are not one.
  const bytes = new Uint8Array(Math.floor((body.length * 3) / 4));
  let out = 0;
  let accumulator = 0;
  let bits = 0;
  for (let index = 0; index < body.length; index += 1) {
    const value = VALUES[body.charCodeAt(index)];
    if (value === undefined) {
      throw new Error(`base64 contains a character that is not in the alphabet at ${index}`);
    }
    accumulator = (accumulator << 6) | value;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes[out] = (accumulator >> bits) & 0xff;
      out += 1;
    }
  }
  if (out !== bytes.length) {
    throw new Error(`base64 decoded to ${out} bytes where ${bytes.length} were expected`);
  }
  return bytes;
}

/** Character code to 6-bit value, built once from the alphabet above. */
const VALUES: (number | undefined)[] = (() => {
  const table: (number | undefined)[] = new Array(128).fill(undefined);
  for (let index = 0; index < ALPHABET.length; index += 1) {
    table[ALPHABET.charCodeAt(index)] = index;
  }
  return table;
})();
