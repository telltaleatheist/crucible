from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
from fastapi import Request, Response
from starlette.background import BackgroundTask

from ... import upstreams
from ...config import Config
from ...engines import chat_admission
from ...errors import ApiError
from ...inflight import InFlight, read_act
from ...sampling import SAMPLING_HEADER, apply_defaults
from ...settle import Settlement
from ..caller import client_agent
from ..proxy import (
    JSON_HEADERS,
    _after_the_stream,
    _caller_gone,
    _chat_body,
    _chat_over,
    _chat_queue_full,
    _engine_unreachable,
    _forward_body,
    _model_not_resident,
    _post_unless_the_caller_leaves,
    _proxy_stream,
    _refuse_an_exited_engine,
    _restore_model_id,
    _sent_across_the_wire,
)
from ..upstream import _forward_to_upstream, _routed_upstream_rows
from ..context import AppContext, Routers


def register(routers: Routers, ctx: AppContext) -> None:
    private, openai = routers.private, routers.openai
    config, residency = ctx.config, ctx.residency

    # --------------------------------------------------------------- openai

    @private.get("/openai/models")
    @openai.get("/models")
    async def openai_models(request: Request) -> dict[str, Any]:
        """The resident model in OpenAI's list shape, plus every routed upstream one.

        `resident_model` rather than `resident`: with a voice on the card there
        is no model to list, and narrator answers no OpenAI route.

        **The upstream rows are the ROUTED ones and not a catalog**
        (PHASE15-HOST.md section 3.4). This route answers *"what may I send as
        `model`"*, and the answer is the resident thing plus whatever the
        operator routed to — the upstream's whole catalog is a different
        question with a different door,
        `POST /v1/settings/upstreams/{name}/test`, and putting it here would
        make a client believe this server had agreed to serve any of them.
        """
        live: Config = request.app.state.config
        resident = residency.resident_model
        data: list[dict[str, Any]] = _routed_upstream_rows(live)
        if resident is None:
            return {"object": "list", "data": data}
        return {
            "object": "list",
            "data": [
                {
                    "id": resident.model_id,
                    "object": "model",
                    "created": int(
                        datetime.fromisoformat(resident.loaded_at).timestamp()
                    ),
                    "owned_by": config.name,
                    # Crucible's id is the contract; this is the name the engine
                    # itself answers to. They differ on mlx-lm, which has no
                    # --served-model-name (crucible/engines/mlx_lm.py).
                    "engine_model_name": resident.engine_model_name,
                    # This entry describes the ENGINE, not the manifest, so both
                    # of these are what was actually loaded. `/v1/models`' row
                    # for the same model reports the manifest's pin, and the two
                    # differ only if somebody edited the manifest while the
                    # engine was up — in which case a client recording what it
                    # talked to wants this one.
                    "revision": resident.revision,
                    "fingerprint": resident.fingerprint,
                    # The context this engine was started with, under OpenAI's
                    # own field name. This is the door Foundry reads — it asks
                    # the OpenAI-shaped listing, not `/v1/models` — and it is the
                    # one that must not lie, because `capFor` subtracts the
                    # prompt from this number to size `max_tokens` and skips the
                    # clamp entirely when it is absent (CLIENT-SURFACES.md
                    # section 6.1).
                    "max_model_len": resident.max_model_len,
                    # What this engine will be sent for a knob the request
                    # leaves out (PHASE2-LLM.md section 9). Here for
                    # `max_model_len`'s reason: Foundry reads the OpenAI-shaped
                    # listing rather than `/v1/models`, and the defaults are a
                    # thing it has to be able to see before it decides what to
                    # send. `null` means the engine's own default.
                    "defaults": resident.defaults.to_dict(),
                },
                *data,
            ],
        }

    @private.post("/openai/chat/completions")
    @openai.post("/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        """Proxied to the resident engine, or forwarded to an upstream.

        PHASE2-LLM.md section 5 is the local half and is unchanged in every
        respect. PHASE15-HOST.md section 3.4 is the other: a `model` of the
        form `<upstream>/<id>` goes to that upstream on the operator's account.

        **The slash is the whole of the test**, and it works because a local
        model id can never contain one — refused at manifest load,
        `manifest_model_id_slash`. One character, one owner, no table.
        """
        raw = await request.body()
        body = _chat_body(raw)
        requested = body.get("model")
        if not isinstance(requested, str) or requested == "":
            raise ApiError(
                400,
                "model_required",
                "a chat request must name a model; this server proxies only to the "
                "model that is resident",
            )
        if upstreams.split_model(requested) is not None:
            return await _forward_to_upstream(
                request, requested, body, client_agent=client_agent(request)
            )
        # A CLEARANCE IS WAITED OUT, AND THE CHAT'S RECORD IS OPENED ATOMICALLY
        # AGAINST ONE BEGINNING (2026-09-24, Briefcase). A chat that arrives
        # while the settlement is SIGTERMing the engine waits for it to finish
        # and is then answered from the settled card — `model_not_resident`,
        # which is true — instead of being proxied to an engine on its way
        # out. The resident check and `inflight.open` are made under the lock
        # the settlement's check-and-claim takes, so a chat that IS admitted
        # is an `InFlight` row the settlement will see and not clear under.
        async with residency.settled_for("a chat request"):
            # The resident MODEL: a voice on the card is not something a chat
            # request can be proxied to, so this door's honest answer is the
            # same `model_not_resident` it gives for an empty card.
            resident = residency.resident_model
            if resident is None or resident.model_id != requested:
                raise _model_not_resident(requested, resident, "a chat request")
            _refuse_an_exited_engine(residency, resident)

            # The manifest's gaps, filled — and the audit of what filled them
            # (PHASE2-LLM.md section 9). `resident.defaults` is what the
            # manifest said at LOAD time, not what it says now, for the same
            # reason `max_model_len` comes off the engine's record.
            applied = apply_defaults(body, resident.defaults)
            sampling_headers = {SAMPLING_HEADER: applied.header()}
            forwarded = _forward_body(raw, applied, resident)
            url = f"{resident.base_url}/v1/chat/completions"
            client: httpx.AsyncClient = request.app.state.http
            inflight: InFlight = request.app.state.inflight
            # Read BEFORE the work starts, so an unknown act is a 400 instead of
            # a completion that ran and was then reported under a name nobody
            # knows.
            act = read_act(request.headers)

            settlement: Settlement = request.app.state.settlement
            chat_over = _chat_over(settlement)

            # A chat is the one piece of accelerator work that took no lane,
            # made no job row and left a record NOWHERE. Tracked so
            # `/v1/activity` can say what this machine is doing; it still gates
            # nothing — see crucible/inflight.py for why taking the lane would
            # have been the wrong fix for the right bug.
            # WHAT THIS ENGINE CAN ACTUALLY HAVE OPEN AT ONCE (2026-09-20).
            # Until today this door admitted everything and
            # `crucible/inflight.py` said, in so many words, that the record
            # gates nothing. For a BATCHING engine that is still exactly right
            # (2026-09-24: vLLM now STATES its batch, `--max-num-seqs`, so it is
            # bounded at that plus one and a client can size to it; the batch
            # is admitted whole, so nothing it could overlap is refused.)
            #
            # It was wrong for a SERIAL one. mlx-lm accepts every connection on
            # a ThreadingHTTPServer and then generates on ONE thread draining
            # ONE queue, so twelve accepted requests are one running and eleven
            # waiting with nothing on the wire saying so. Foundry's clean pass
            # died there on 2026-09-20: 12 in flight, a 300 s client deadline, a
            # request that had not started when it passed, the pass dead at
            # block 352 of 940.
            #
            # (CORRECTED 2026-09-24: mlx-lm is not serial — that one thread runs
            # a `BatchGenerator` `--decode-concurrency` wide. The door was right
            # to bound it and wrong about the width; the width is now read off
            # the resident engine's argv, `engines/mlx_lm.py`.)
            #
            # A refusal a client can act on beats a socket that goes quiet. The
            # limit is the engine's own measured concurrency plus one (see
            # `engines.chat_admission`), the wait is the median of what
            # completions on this engine have actually been taking, and a server
            # that has finished none states no `Retry-After` rather than
            # inventing one — `_rate_limited`'s rule, applied to a number of
            # our own.
            limit, limit_basis = chat_admission(resident.engine, resident.engine_args)
            if limit is not None and len(inflight) >= limit:
                wait = inflight.retry_after()
                return _chat_queue_full(
                    resident=resident, limit=limit, basis=limit_basis, wait=wait
                )

            entry = inflight.open(
                act=act, model=resident.model_id, client=client_agent(request)
            )
        try:
            if body.get("stream") is True:
                # THE RELAY OUTLIVES THIS HANDLER, so the record and the
                # settlement belong to the relay's end and not to this `return`.
                # Closing them here would count a streamed completion as finished
                # the moment its first byte was ready — and since 2026-09-14 that
                # would clear the card out from under a stream still producing
                # tokens.
                return await _proxy_stream(
                    client,
                    url,
                    forwarded,
                    resident,
                    sampling_headers,
                    when_relayed=_after_the_stream(inflight, entry, chat_over),
                )
            try:
                upstream = await _sent_across_the_wire(
                    lambda: _post_unless_the_caller_leaves(
                        client, url, forwarded, request, JSON_HEADERS
                    ),
                    where=f"the engine serving {resident.model_id!r}",
                )
            except httpx.HTTPError as exc:
                raise _engine_unreachable(resident, exc) from None
            if upstream is None:
                response = _caller_gone(resident)
            else:
                content = upstream.content
                if upstream.status_code == 200:
                    content = _restore_model_id(content, resident)
                response = Response(
                    content=content,
                    status_code=upstream.status_code,
                    media_type=upstream.headers.get(
                        "content-type", "application/json"
                    ),
                    # On the engine's own refusal too: a 400 whose `max_tokens`
                    # came from the manifest is a 400 the manifest caused, and the
                    # reader needs that on the response that carries it.
                    headers=sampling_headers,
                )
            inflight.close(entry)
            # AFTER THE ANSWER IS WRITTEN, not before it. Starlette runs a
            # response's background task once the body has gone out, so a client
            # gets its completion at the speed the engine produced it and the
            # card is cleared behind it. A settlement can wait on a SIGTERM for
            # as long as three minutes, and no answer should be held for that.
            response.background = BackgroundTask(chat_over)
            return response
        except BaseException:
            # A refusal, a disconnect, an engine that died: the record must not
            # outlive the request, and the card is as free now as it would have
            # been had this succeeded.
            inflight.close(entry)
            await chat_over()
            raise
