from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import Request

from ... import API_VERSION, VERSION, accelerator
from ...backend import CUDA_LINUX
from ...errors import ApiError
from ...inflight import InFlight
from ...jobs.queue import JobStore
from ...leases import Leases
from ...settle import Settlement
from ...ttsstream import StreamManager
from ..proxy import _chat_limit_of
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend, residency = ctx.config, ctx.backend, ctx.residency

    # ----------------------------------------------------------- accelerator

    @private.get("/accelerator")
    async def accelerator_state(request: Request) -> dict[str, Any]:
        """What is on the card right now, and which of it is Crucible's.

        PHASE4-AUDIO.md section 5. This is the same `nvidia-smi
        --query-compute-apps` the load guard runs, plus the free/total figures,
        plus the resident set, plus a flag saying which holders are this server's
        own processes — and it exists because BookForge arbitrates the GPU three
        incompatible ways at once (a queue slot, an in-process mutex whose
        timeout *proceeds without the lock*, and nothing at all for the hosted
        page reader), on top of a lock file with no producer inside the app. One
        call here answers the question all three were guessing at.

        **It never evicts anybody, ever.** It reports, and that is the whole of
        it. The rule is PHASE2-LLM.md section 4's and it does not soften because
        more job types now depend on the answer.

        It is private like every other route here: the bearer token and the
        version header, in that order. A probe of somebody's hardware is not
        public information, and `GET /v1/ping` already exists for "is this a
        Crucible".
        """
        # nvidia-smi is a subprocess and takes tens of milliseconds; off the
        # event loop, or a poll of this route stalls every job's event stream.
        try:
            state = await asyncio.to_thread(
                accelerator.read_state, backend.kind, config.desktop_allowance_bytes
            )
        except accelerator.ProbeError as exc:
            # 503 and not 409: nothing was asked for and refused, the server
            # simply cannot see its own card at the moment. A client polling for
            # a free GPU must read this as "ask again", never as "it is free" —
            # which is why the probe raises rather than returning zeroes.
            raise ApiError(
                503,
                "accelerator_unreadable",
                f"this server cannot read its accelerator: {exc}",
            ) from None

        owned = residency.owned_pids()
        holders = [
            {
                "pid": app.pid,
                "name": app.name,
                # None where the driver will not say (WDDM, permissions). That is
                # not zero and must not be rendered as zero.
                "bytes": app.used_bytes,
                "owned_by_crucible": app.pid in owned,
            }
            for app in state.compute_apps
        ]
        resident = residency.resident
        return {
            "backend": state.backend,
            "gpu": {
                "vendor": backend.gpu.vendor,
                "name": backend.gpu.name,
                # The live figure from the probe, not the one detection recorded
                # at start-up. They agree on a real host; where they would not,
                # the live one is the one a caller is about to make a decision on.
                "total_bytes": state.total_bytes,
            },
            "free_bytes": state.free_bytes,
            "used_bytes": state.used_bytes,
            "desktop_allowance_bytes": config.desktop_allowance_bytes,
            # VRAM in use that no listed compute app accounts for, past the
            # declared desktop allowance. Under WSL2 the driver shim answers the
            # compute-app query with an EMPTY LIST even while a process inside
            # that same VM holds 17 GB (measured on Owen's PC, 2026-09-12), so on
            # that host this number is the only honest report of the card being
            # busy and `holders` will be misleadingly empty. Null on mlx-darwin,
            # where "used unified memory" is the OS doing its job and attributing
            # it to compute processes is not a question vm_stat can answer.
            "unattributed_bytes": (
                accelerator.unattributed_bytes(state, config.desktop_allowance_bytes)
                if state.backend == CUDA_LINUX
                else None
            ),
            "resident": (
                None
                if resident is None
                else {
                    # `kind` is the family of thing that is resident, not the job
                    # type that put it there — and it is ASKED rather than
                    # assumed. This said `"llm"` and read `resident.model_id`
                    # until 2026-09-13, which was true while a model was the only
                    # thing a card could hold and became a 500 the moment
                    # PHASE3-TTS.md section 5's generalised residency landed: a
                    # `ResidentVoice` has a `voice_id` and an `id`, and no
                    # `model_id` at all. `model_rows()` had already learned to ask
                    # for `resident_model`; this route had not caught up, so the
                    # route whose whole job is to say what is on the card was the
                    # one that could not say a voice was.
                    #
                    # `id` and `kind` are what every Resident has in common, by
                    # design: phase 4's aligner is a third kind and needs no
                    # change here.
                    "kind": resident.kind,
                    "id": resident.id,
                    "since": resident.loaded_at,
                    "memory_bytes_estimate": resident.memory_bytes_estimate,
                }
            ),
            "holders": holders,
            "detail": state.detail,
        }

    def _activity_row(store: JobStore, job: Any) -> dict[str, Any]:
        """One job, as a bench reads it. Never its params: a chat prompt or a
        chapter of a book is not something a whole-server read should spray at
        anyone holding the token."""
        return {
            "job_id": job.id,
            "type": job.type,
            "model": job.model,
            "status": job.status,
            "position": store.position(job),
            "progress": job.progress,
            "message": job.message,
            "created": job.created,
            "started": job.started,
            "client": job.client,
        }

    @private.get("/activity")
    async def activity(request: Request, accelerator_probe: bool = False) -> dict[str, Any]:
        """What is on this server and how far along — one read, no job id.

        PHASE7-LANES.md section 5. Owen, 2026-09-13: *"Crucible will have to have
        an api endpoint that will report what's on it and its progress so
        Bookforge can hit that endpoint and fill that gpu slot with that data."*

        WHY THIS IS A POLL AND NOT THE SSE IT ALREADY HAS. Per-job events are
        push, fine-grained and exactly right for the step that owns a job. This
        answers a different question, asked by a bench widget that owns no job
        and may never own one: *what is this machine doing?* Opening a stream per
        job per server to render one line of text is the wrong shape. The two do
        not compete — the step reads the stream, the bench reads this.

        IT REPORTS AND NOTHING ELSE. It does not admit, reserve, claim or lock. A
        client that reads "free" and submits is racing every other client, and
        that race is settled at the door: `POST /v1/jobs` admits one and refuses
        the other `server_busy`, naming the winner (ARCHITECTURE.md section 3).
        The loser has lost nothing but a round trip, because it never gave up
        ownership of its own queue — which is the point of the ruling. A
        reservation here would be a second place to arbitrate, and a stale one.

        **So this route is a bench display and a preflight, never admission.** It
        is the honest answer to "how long until that finishes"; it is not
        permission to submit, and a client must be able to be refused after
        reading it. Only `POST /v1/jobs` can say yes.

        THE PROBE IS OPT-IN, and that is the one design decision in this route.
        `nvidia-smi` is a subprocess costing tens of milliseconds, and a bench
        polling three servers every few seconds would spawn one per server per
        tick forever to render a number nobody is reading. `resident` below
        already says what is loaded and roughly what it costs, in memory, for
        free. A caller that genuinely wants the live figure asks for it with
        `?accelerator_probe=true` and pays for it; `GET /v1/accelerator` remains
        the full answer.
        """
        store: JobStore = request.app.state.store
        streams: StreamManager = request.app.state.streams
        inflight: InFlight = request.app.state.inflight
        leases: Leases = request.app.state.leases
        running = store.running
        queued = store.queued()
        resident = residency.resident
        session = streams.session
        lease = leases.current()
        chat_limit, chat_limit_basis = _chat_limit_of(residency)
        settlement: Settlement = request.app.state.settlement
        held = settlement.held_by()
        unclaimed = settlement.unheld_since()

        body: dict[str, Any] = {
            "server": {
                "name": config.name,
                "version": VERSION,
                "api_version": API_VERSION,
                "backend": backend.kind,
                "uptime_s": round(time.monotonic() - request.app.state.started_at, 3),
            },
            "resident": (
                None
                if resident is None
                else {
                    "kind": resident.kind,
                    "id": resident.id,
                    "since": resident.loaded_at,
                    "memory_bytes_estimate": resident.memory_bytes_estimate,
                    # NOT SERVING, said here rather than discovered by a chat
                    # that hangs: the code the resident model's engine exited
                    # with, null while it runs (2026-09-26, ContentStudio's
                    # dead 27B read as healthy for 17-20 minutes).
                    "engine_exit_code": residency.engine_exit_code,
                    # WHICH CLIP, for a zero-shot voice. `zeroshot` is ONE
                    # voice id and any number of recordings — BookForge keeps
                    # its clips in a userData directory and the extension
                    # keeps its own in the browser — so the id alone is two
                    # clients each assuming the resident one is theirs. Null
                    # for every other kind and for a model, and always
                    # present: an absent key would mean "this build does not
                    # say" (PHASE3-TTS.md section 5).
                    "reference": getattr(resident, "reference", None),
                    # WHAT HOLDS IT, and since when it has been held by
                    # nothing (2026-09-20). `resident` says what is on the
                    # card; these say whether anybody is coming back for it.
                    #
                    # Both read `crucible/settle.py`, which is the ONE owner of
                    # "what holds the card" — four facts, one function. Deriving
                    # them again here would be the one-fact-two-owners shape
                    # `docs/ARCHITECTURE.md` says this repo keeps finding, and
                    # the copy would drift the first time a fifth kind of
                    # holder arrived.
                    #
                    # `held_by` null with `resident` set is the stranded card: a
                    # `load-model` that succeeded and was never claimed, or a
                    # lease that lapsed with nothing asking again. Neither is
                    # hypothetical — the first happened on this server at
                    # 18:29:37Z on 2026-09-20, when a runner was stopped one
                    # second after its load reached `done`.
                    #
                    # REPORTING ONLY. Nothing here unloads anything and there is
                    # no timer: what should be DONE about a card nobody holds is
                    # a ruling (`docs/BUG-HUNT-2026-09-20.md` §F.8). This is the
                    # half that needs no ruling, because a reconciler on either
                    # side cannot act on a state the server will not say.
                    "held_by": (None if held is None else held.to_dict()),
                    "unclaimed_since": (
                        None if unclaimed is None else unclaimed.isoformat()
                    ),
                }
            ),
            # WHAT WAS TOLD TO GO AND HAS NOT (ledger R13, Owen's ruling
            # 2026-09-18). `resident` above says what may be USED; this says
            # what is still ON THE CARD after a stop that was asked for and
            # never confirmed (`crucible/residency.py`'s dying slot). The two
            # are never both set.
            #
            # It is on the bench read because it is a state a human has to
            # end: `engines/base.py` never SIGKILLs — a killed CUDA process
            # wedges WSL2 until Windows reboots — so nothing in Crucible will
            # clear this, and every load, the claim and the streaming door go
            # on refusing `engine_still_stopping` until somebody stops those
            # pids. `accepts_work` below is untouched and still true, and
            # truthfully so — the lane is free and an `echo` or a render
            # against nothing resident is still admitted. What was missing
            # was any way for a bench to explain the load that comes back
            # `engine_still_stopping` off a server that looks idle.
            "stopping": (
                None if residency.stopping is None else residency.stopping.to_dict()
            ),
            # `warming` is neither running nor queued and a bench that ignored it
            # would draw an idle machine that is in fact spending two minutes
            # loading a model. It is the reason a slot is unavailable, so it is
            # reported where the slot is.
            "warming": residency.warming,
            # WHO HOLDS NARRATOR'S WIRE, WHICH IS NOT THE SAME QUESTION AS THE
            # LANE. `refuse_if_claimed`'s docstring is the long version: a
            # streaming session holds the resident engine *without* occupying the
            # lane, so `slots` below can say this server is free while the card is
            # not. Until this field existed, a bench polling for a free machine
            # read `busy: 0` and `running: []` **while the browser extension was
            # streaming from it**, submitted, and was refused `engine_in_use`
            # after the round trip. The refusal was right; the display was a lie,
            # and it lied in the one direction that matters (R3: nothing is ever
            # told "maybe" — and "free" when it is not is worse than "maybe").
            #
            # Reported as its own field rather than folded into `slots` because it
            # is a different fact with a different owner: the lane belongs to
            # `JobStore`, the claim belongs to `Residency`. Folding them would
            # give the composite a third owner and lose which one said no.
            "claim": (
                None
                if residency.claimed_by is None
                else {"held_by": residency.claimed_by}
            ),
            # THE OTHER KIND OF WORK. Three BookForge surfaces stream rather than
            # queue — the streaming page, the correct-sentences/re-roll page and
            # the browser extension — and they claim a server for as long as a
            # reader keeps reading. `progress` is null and always will be: see
            # `StreamSession.progress_report` for why a session has no
            # denominator and what is counted instead.
            "streaming": (
                None
                if session is None
                else {
                    "session_id": session.id,
                    "voice": session.voice,
                    "language": session.language,
                    "narrator_engine": session.narrator_engine,
                    "since": session.opened_at,
                    "client": session.client,
                    "progress": None,
                    **session.progress_report(),
                }
            ),
            # THE THIRD KIND OF WORK, and the one that was invisible longest. A
            # chat completion takes no lane, makes no job and left no record, so
            # a server grinding through a 27B translation reported `running: []`
            # and read as idle. It is counted here and it still gates nothing:
            # a vLLM engine BATCHES, so two passes on one resident model really
            # do run at once, and taking the lane to fix a reporting bug would
            # have serialised work the engine exists to overlap.
            #
            # `act` is the client's word (the `X-Crucible-Act` header, validated
            # against the capability classes). Null means it did not say, and
            # this server never guesses one: it cannot tell a simplify from a
            # translate, since both are a chat against the same 27B and the only
            # difference is a prompt it does not own.
            # `max_in_flight` is what THIS engine's door will admit at once,
            # and `max_in_flight_basis` is where that number came from, so a
            # client can size its own pool from the server instead of guessing
            # and discovering the answer as a starved socket. Null for an engine
            # that states no concurrency — the door then bounds nothing, which
            # is mlx-vlm today — and null when no model is
            # resident, because the limit belongs to the engine and there is no
            # engine to ask.
            "chat": {
                "in_flight": len(inflight),
                "max_in_flight": chat_limit,
                "max_in_flight_basis": chat_limit_basis,
                "rows": inflight.rows(),
            },
            # WHO CHANGED THIS SERVER'S SETTINGS, AND WHEN (PHASE15-HOST.md
            # section 3.2). Two apps and the operator page can all write the
            # same engine, so "why is translate suddenly on Anthropic" needs an
            # answer that is not "read three apps' logs". Newest first, the
            # last `settings.HISTORY_LIMIT` of them, in memory.
            #
            # **The field paths, never the values of an upstream.** A route's
            # model id is recorded because it is not a secret; an upstream
            # entry records `set` or `removed` and nothing more, which is the
            # whole of "a key appears in no activity record".
            "settings": {"writes": request.app.state.settings_history.rows()},
            # WHAT WAS DELETED, AND BY WHOM (PHASE15-HOST.md 3.5a). The host's
            # weights migration deletes a Windows copy once the guest has its
            # own, and a person looking at a machine with less on it than they
            # remember needs an answer that is not "read the host's log".
            # Newest first, in memory, the last `Removals.LIMIT`.
            "catalog": {"removals": request.app.state.removals.rows()},
            # THE INTENTION BEHIND THE CHATS, which no amount of looking at this
            # server could infer. A chat holds nothing and is over in seconds, so
            # between two blocks of a 2000-block translation this machine is idle
            # by every other measure here — and a `load-voice` submitted in that
            # gap used to evict the translator (crucible/leases.py). While this
            # is non-null, the thing on the card cannot be moved.
            #
            # It does NOT change `accepts_work` below. A lease is not a
            # reservation: this server will still take a job that does not need
            # the card's contents to change, and admission is still the door's.
            #
            # No `subject` field: a lease is only ever on the resident thing, and
            # `resident.id` above is already that fact's owner (R1). `kind` IS
            # carried, because the same six fields are a `409 leased`'s details —
            # a document with no `resident` beside it — and the kind is what says
            # which jobs the refusal covers.
            "lease": None if lease is None else lease.to_dict(),
            "slots": {
                # ONE LANE TODAY, and it is named rather than counted so the
                # ancillary lane (PHASE7-LANES.md section 3) can appear beside it
                # without changing this one's meaning. A key that is absent means
                # this build has no such lane — never that the lane is idle.
                #
                # `busy` counts THE LANE and nothing else — a stream does not take
                # it, and saying otherwise would redefine the lane to mean "the
                # card", which is `claim`'s job above. What a caller actually
                # wants before submitting is `accepts_work`, which is the
                # composition, derived here once so that three benches do not each
                # invent their own and disagree.
                "accelerated": {
                    "busy": 0 if running is None else 1,
                    "of": 1,
                    "queue_depth": store.queue_depth,
                    # Derived, never stored. Still not a reservation: a client
                    # that reads true and submits is racing every other client,
                    # and that race is settled at the door. See this route's
                    # "IT REPORTS AND NOTHING ELSE" note.
                    #
                    # `chat` is deliberately NOT a term here. A chat in flight
                    # does not stop this server taking a job, because the engine
                    # batches — adding it would turn an honest display into a
                    # false refusal and serialise work that overlaps today.
                    "accepts_work": running is None and residency.claimed_by is None,
                },
            },
            "running": [] if running is None else [_activity_row(store, running)],
            "queued": [_activity_row(store, job) for job in queued],
        }

        if accelerator_probe:
            try:
                state = await asyncio.to_thread(
                    accelerator.read_state, backend.kind, config.desktop_allowance_bytes
                )
            except accelerator.ProbeError as exc:
                # NOT a 503 for the whole route, unlike `/v1/accelerator`. The
                # caller asked for the bench and additionally for a probe; a card
                # the driver will not talk about right now must not blank out the
                # progress of a render that is plainly still going. The failure is
                # named in place and everything else stands.
                body["accelerator"] = {"error": f"{exc}"}
            else:
                body["accelerator"] = {
                    "total_bytes": state.total_bytes,
                    "free_bytes": state.free_bytes,
                    "used_bytes": state.used_bytes,
                    "unattributed_bytes": (
                        accelerator.unattributed_bytes(
                            state, config.desktop_allowance_bytes
                        )
                        if state.backend == CUDA_LINUX
                        else None
                    ),
                }
        return body
