import { RELEASE_REPO } from './release.js';
import { BootstrapRefusal } from './errors.js';

/** GitHub's own pointer at the promoted release. */
export const LATEST_RELEASE_URL = `https://api.github.com/repos/${RELEASE_REPO}/releases/latest`;

const VERSION_RE = /^v?(\d+)\.(\d+)\.(\d+)/;

function unreadable(why: string, detail?: string): BootstrapRefusal {
  return new BootstrapRefusal(
    'release_channel_unreadable',
    `could not read the release channel at ${LATEST_RELEASE_URL}: ${why}`,
    detail === undefined ? {} : { detail },
  );
}

/** The version `releases/latest` names, from the document it answers with. */
export function parseLatestRelease(text: string, url: string): string {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw unreadable(`it is not JSON (${(err as Error).message})`, text.trim().slice(0, 200));
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw unreadable(`${url} answered something that is not a release document`);
  }
  const tag = (parsed as Record<string, unknown>)['tag_name'];
  if (typeof tag !== 'string' || !VERSION_RE.test(tag)) {
    throw unreadable(`its tag_name is ${JSON.stringify(tag)}, which is not a Crucible version`);
  }
  return tag.replace(/^v/, '');
}

/** Ask the channel what its latest release is. */
export async function latestRelease(fetchImpl: typeof globalThis.fetch = globalThis.fetch): Promise<string> {
  let response: Awaited<ReturnType<typeof globalThis.fetch>>;
  try {
    response = await fetchImpl(LATEST_RELEASE_URL, {
      headers: { accept: 'application/vnd.github+json' },
    });
  } catch (err) {
    throw unreadable((err as Error).message, (err as Error).stack);
  }
  const body = await response.text();
  if (!response.ok) throw unreadable(`HTTP ${response.status}`, body.trim().slice(0, 200));
  return parseLatestRelease(body, LATEST_RELEASE_URL);
}

/**
 * Order two releases numerically: negative when `a` is older, 0 when equal, positive when newer.
 */
export function compareReleases(a: string, b: string): number {
  const left = VERSION_RE.exec(a.trim());
  const right = VERSION_RE.exec(b.trim());
  if (left === null || right === null) {
    throw new BootstrapRefusal(
      'release_channel_unreadable',
      `${JSON.stringify(left === null ? a : b)} is not a Crucible version, so it cannot be compared with `
        + `${JSON.stringify(left === null ? b : a)}`,
    );
  }
  for (let index = 1; index <= 3; index += 1) {
    const difference = Number(left[index]) - Number(right[index]);
    if (difference !== 0) return difference;
  }
  return 0;
}
