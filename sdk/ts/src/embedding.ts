import { decodeBase64 } from './base64.js';
import { CrucibleConfigError } from './errors.js';
import type { EmbedEncoding } from './types.js';

/** One IEEE 754 half-precision value, from its 16 bits. */
function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

/**
 * One vector of an {@link EmbedResponse}, as floats: a `float` vector as it is, a `base64` one read
 * as float32 little-endian, a `base64_float16` one as IEEE half little-endian.
 */
export function decodeEmbedding(vector: readonly number[] | string, encoding: EmbedEncoding): Float32Array {
  if (encoding === 'float') {
    if (typeof vector === 'string') {
      throw new CrucibleConfigError('vector', 'is a string, and a `float` vector is an array of numbers');
    }
    return Float32Array.from(vector);
  }
  if (typeof vector !== 'string') {
    throw new CrucibleConfigError('vector', `is not a base64 string, which a \`${encoding}\` vector is`);
  }
  const bytes = decodeBase64(vector);
  const width = encoding === 'base64' ? 4 : 2;
  if (bytes.length % width !== 0) {
    throw new CrucibleConfigError('vector', `decodes to ${bytes.length} bytes, not whole ${width}-byte values`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.length / width);
  for (let index = 0; index < out.length; index += 1) {
    out[index] =
      width === 4 ? view.getFloat32(index * 4, true) : halfToFloat(view.getUint16(index * 2, true));
  }
  return out;
}
