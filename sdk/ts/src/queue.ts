import { CrucibleConfigError } from './errors.js';
import type { QueueChoice } from './types.js';

/** The shortest wait the server's queue accepts, in seconds. */
export const MIN_MAX_WAIT_S = 10;
/** The longest wait the server's queue accepts (a day), in seconds. */
export const MAX_MAX_WAIT_S = 86_400;

/** The `queue` member a request sends for a {@link QueueChoice}, or null to send none. */
export function queuePayload(choice: QueueChoice | undefined): Record<string, number> | null {
  if (choice === undefined || choice === false) return null;
  if (choice === true) return {};
  if (typeof choice !== 'object' || choice === null) {
    throw new CrucibleConfigError('queue', 'must be true, false or {maxWaitS}');
  }
  if (choice.maxWaitS === undefined) return {};
  return { max_wait_s: requireSeconds(choice.maxWaitS, 'queue.maxWaitS', 'an hour') };
}

/**
 * A whole number of seconds the server's queue accepts (10..86400), refused before anything is
 * sent otherwise. `fallback` names the server's own default, for the sentence.
 */
export function requireSeconds(value: unknown, option: string, fallback: string): number {
  if (
    typeof value !== 'number' ||
    !Number.isInteger(value) ||
    value < MIN_MAX_WAIT_S ||
    value > MAX_MAX_WAIT_S
  ) {
    throw new CrucibleConfigError(
      option,
      `is ${String(value)}; the server takes a whole number of seconds from ` +
        `${MIN_MAX_WAIT_S} to ${MAX_MAX_WAIT_S}. Leave it out for the server's default (${fallback}).`,
    );
  }
  return value;
}
