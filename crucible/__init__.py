"""Crucible — one inference server, many client apps.

`VERSION` is the build (semver). `API_VERSION` is the contract clients check; it is
bumped only on a breaking change to the HTTP surface. The two are deliberately
separate (DESIGN.md section 8).

`API_HEADER` is the header that carries `API_VERSION`. It lives HERE and not in
`crucible/api.py`, which is where it was written, because `crucible/peer.py`'s
orchestrator half has to send it and must not import the server: that half runs
inside the Windows tray, which starts at login and must not drag FastAPI,
uvicorn or httpx in behind it. One spelling, in the one module both sides can
import for free.

`KEEP_ALIVE_SECONDS` is here for the same reason: there are two uvicorn doors
(`cli.py:cmd_serve` and `host/child_lifecycle.py:run_owned_server`), the second
of them lives in the tray's own package and must not import the first, and a
keep-alive that differed between them would be the defect below on one door
only.
"""

VERSION = "1.0.36"
API_VERSION = 1
API_HEADER = "X-Crucible-Api"
#: The header a client NAMES ITSELF in, read by `_client_agent` in preference to
#: the User-Agent.
#:
#: It exists because a browser cannot set a User-Agent — it is a forbidden
#: header name, so the SDK's `clientName` is silently dropped by Chrome and the
#: BookForge Reader extension's jobs were recorded under a 120-character
#: `Mozilla/5.0 …` string. Here beside `API_HEADER` rather than in `api.py`,
#: for the reason that one is here: the SDK and the server must spell a header
#: the same way, and a spelling with two homes eventually has two spellings.
CLIENT_NAME_HEADER = "X-Crucible-Client"

#: HOW LONG AN IDLE HTTP CONNECTION IS HELD OPEN, for every uvicorn this repo
#: starts. **The requirement is that the SERVER always outlives the CLIENT
#: pool's idle window**, because whoever closes first decides who gets the
#: error: a socket the server is closing while the client writes the next
#: request onto it is an `ECONNRESET` the client cannot distinguish from a dead
#: server.
#:
#: MEASURED, 2026-09-20. uvicorn's default `timeout_keep_alive` is **5 s** and
#: neither door passed one. Node's `fetch` (undici) keeps an idle pooled
#: connection about **4 s**. Those two numbers are close enough that a request
#: a few seconds after the last one lands in the gap, and it did, four times:
#: align's first `GET /v1/info` after a render's last artifact fetch failed
#: `read ECONNRESET` on Sep 18 and Sep 19 against the PC and on Sep 20 at 00:57
#: against the Mac on 127.0.0.1 — with `serve.log` showing no gap and
#: `crucible api ping` answering before and after. The server was up the whole
#: time; the socket was not.
#:
#: **75 is chosen from the ordering and not from folklore.** What has to be
#: true is `server >> client`, and the client window that produced the four
#: failures is undici's ~4 s — so anything in the tens of seconds satisfies it
#: with a margin no clock skew or GC pause can close. 75 is taken as the number
#: that also sits above the 60 s a proxy or tunnel is conventionally configured
#: to idle out at, so a Crucible behind one is not the side that closes first
#: either. It is not a measured number and must not be quoted as one; what is
#: measured is the 4 s and the 5 s above. The cost is one open file descriptor
#: per idle client, on a server whose clients are counted in ones.
#:
#: **This is one half of the fix and not the whole of it.** A server can always
#: close a connection — restarts, a proxy in the middle, a network that drops —
#: so the SDK retries an idempotent request once on a reset that arrives before
#: any response byte (`sdk/ts/src/client.ts`). Raising this number alone would
#: make the failure rare instead of impossible, which is how it survived from
#: Sep 18 to Sep 20.
KEEP_ALIVE_SECONDS = 75

__all__ = ["VERSION", "API_VERSION", "API_HEADER", "CLIENT_NAME_HEADER",
           "KEEP_ALIVE_SECONDS"]
