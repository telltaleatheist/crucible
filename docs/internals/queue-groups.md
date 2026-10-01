# Queue groups: a proposal to replace leases

Status: **proposal, 2026-10-01, not built.** Owen: *"we might be able to get rid of leases if we
have the queue. we can create queue groups that guarantee queue items come one after another."*

## What a lease is for, and where it falls short

A lease says "keep this subject on the card while my run goes on." Apps take one because their
work is a sequence of requests they cannot know in advance (BookForge chapters, Briefcase
scoring a video): each request depends on the last answer, the run lasts minutes to hours, and
losing the card between two steps means a reload or, worse, someone else's job in the middle.

Its shortcomings:

- It pins **one subject**. Briefcase's run needs ASR, then the LLM, then the LLM again;
  a lease on the LLM cannot cover the ASR step, so another client can slip in between.
- It is a **timer plus a heartbeat**. A client that stalls loses the card; a client that
  dies holds it until the TTL runs out.
- It sits **beside the queue**, not in it. A waiting lease now joins the line (1.0.74), but a
  granted one is a separate mechanism with its own refusals (`leased`), its own expiry, and
  its own rules in the settlement.
- The operator sees "a lease", not the run: what it is doing, how far along, what is next.

## The group

A **group** is a client's claim on the lane for a sequence of requests. While it is open,
its items run one after another with nothing from anyone else in between, and what they
leave on the card stays there for the next item.

```
POST   /v1/queue/groups                 {"act": "score", "subjects": ["qwen3.5-9b"],
                                         "idle_s": 120, "queue": {"max_wait_s": 3600}}
  -> 201 {"group_id": "grp-…", "opened_at": …}       (held open while it waits, like a lease)
POST   /v1/jobs | /v1/decide | /v1/openai/chat/completions | …
       with header  X-Crucible-Group: grp-…          (an item of the group)
POST   /v1/queue/groups/{id}/touch                   (still here; only for long client-side gaps)
DELETE /v1/queue/groups/{id}                         (done; the card is settled)
```

- **Opening it waits in the line** like any queued item (kind `group`). At the front it is
  granted; `subjects`, if given, are loaded or kept resident for it (the first one now, the
  others when an item needs them).
- **Items skip the line.** A request carrying the group's header is admitted at once, ahead of
  everything waiting, and is never refused `leased` or `server_busy` on account of other
  clients. Jobs, chats and decisions are all items; an item can load a different subject (ASR,
  then the LLM) without anyone getting in between.
- **Nothing else runs on the lane while it is open.** Other clients' queued work waits;
  their unqueued jobs are refused `server_busy` as they are behind a busy lane today. Chats from
  other clients to the resident model are the one open question (below).
- **No heartbeat.** Every item counts as presence. The group closes itself after `idle_s` with
  no item, no touch and nothing in flight (reason `idle`), or when the client closes it. An
  operator can end it from the desktop Queue (reason `operator`). A maximum hold
  (`max_hold_s`, a server limit) stops one client keeping the machine for a day.
- **The settlement treats an open group as a holder**, exactly as it treats a lease today.
  When the group closes, the card is settled (unloaded unless something else holds it).
- **Visible as one row** in `GET /v1/queue` and the desktop app: the client, the act, how long
  it has been open, how many items it has run, what is in flight.

## Leases become a special case, then go

1. Build groups on the line that exists (the pump, `crucible/callqueue.py`, the settlement's
   holder rule).
2. Re-express a lease as a group: `POST /v1/models/{id}/lease` opens a group with
   `subjects: [id]`, `idle_s` = `ttl_seconds`, and the lease heartbeat touches it. Old
   clients keep working unchanged; the `leased` refusal becomes "a group holds the lane".
3. The SDK gains `group({act, subjects, idleS})` returning a handle whose methods (`chat`,
   `decide`, `submit`, …) send the header, and `close()`. `lease()` stays as a thin wrapper.
4. Once BookForge, Briefcase and Foundry run on groups, the lease routes are retired.

## Decisions for Owen

1. **Other clients' chats during a group.** Block them (strict "nothing in between"), or let
   unqueued chats to the resident model through (they do not move the card, but they take
   engine slots and GPU time). Proposed: block queued ones, let unqueued ones through until
   leases are retired, then block both.
2. **Maximum hold.** Proposed 4 h by default, set in config; a group past it is ended
   `max_hold` and its client told.
3. **One group at a time per server** (proposed), with other clients' groups waiting in line.
