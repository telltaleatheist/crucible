/**
 * The release channel — the ONE thing that says which Crucible is "latest".
 *
 * docs/INSTALL-UNINSTALL.md §6.5.1. Every cut is created
 * `--prerelease --latest=false` by `scripts/release.sh` and becomes the channel's
 * latest only when `scripts/promote_release.py --publish` runs
 * `gh release edit --prerelease=false --latest=true`, after its packs, its assets
 * and a fresh-install smoke have been verified. So the promoted release is what
 * `releases/latest` answers, and that is the pointer — NOT `releases?per_page=1`,
 * which answers "the newest tag created" and is therefore an unverified candidate
 * on every day between a cut and its promotion.
 *
 * There is NO fallback under this (§6.5.2). A channel that cannot be read is
 * `release_channel_unreadable` by name, carrying what failed; it is never a
 * reason to install the version this package happens to have been built at,
 * because that is exactly how an app came to install 1.0.1 over a running 1.0.2.
 * The one override is an operator naming an exact release — which pins WHICH
 * release, and still downloads it from GitHub. There is no offline install.
 */
import { RELEASE_REPO } from './envpacks.js';
import { BootstrapRefusal } from './errors.js';

/** GitHub's own pointer at the promoted release. One URL, spelled once. */
export const LATEST_RELEASE_URL = `https://api.github.com/repos/${RELEASE_REPO}/releases/latest`;

/** A release version — three numbers. The tag is the same thing with a `v`. */
const VERSION_RE = /^v?(\d+)\.(\d+)\.(\d+)/;

function unreadable(why: string, detail?: string): BootstrapRefusal {
  return new BootstrapRefusal(
    'release_channel_unreadable',
    `could not read the release channel at ${LATEST_RELEASE_URL}: ${why}`,
    detail === undefined ? {} : { detail },
  );
}

/**
 * The version `releases/latest` names, from the document it answers with.
 *
 * Separate from the fetch so that the parsing is testable without a network and
 * so the refusal for "GitHub answered something that is not a release" is the
 * same one as for "GitHub did not answer": a reader who cannot learn the
 * channel's latest has learnt nothing either way, and both ways must stop.
 */
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

/**
 * Ask the channel what its latest release is.
 *
 * `fetchImpl` is a parameter because an app hands its own fetch in and a test
 * hands a scripted one; there is no second behaviour behind it.
 */
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
 * Order two releases: negative when `a` is older, 0 when they are the same
 * release, positive when `a` is newer. A tag (`v1.0.2`) and a version (`1.0.2`)
 * are the same release.
 *
 * NUMBER BY NUMBER, because a string comparison puts 1.0.10 before 1.0.2 and
 * the whole point of every gate that calls this is to know which of two
 * versions is older. A string that is not three numbers is refused rather than
 * sorted to one end: "older" is not a question that has an answer about it.
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
