/**
 * Base64 for inline job inputs.
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
