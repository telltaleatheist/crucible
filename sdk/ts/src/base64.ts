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

export function decodeBase64(text: string): Uint8Array {
  const body = text.endsWith('==')
    ? text.slice(0, -2)
    : text.endsWith('=')
      ? text.slice(0, -1)
      : text;
  if (text.length % 4 !== 0 || body.length % 4 === 1) {
    throw new Error(`base64 of length ${text.length} is not a whole number of groups`);
  }

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

const VALUES: (number | undefined)[] = (() => {
  const table: (number | undefined)[] = new Array(128).fill(undefined);
  for (let index = 0; index < ALPHABET.length; index += 1) {
    table[ALPHABET.charCodeAt(index)] = index;
  }
  return table;
})();
