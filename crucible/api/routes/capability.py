from __future__ import annotations

from typing import Any

from fastapi import Request

from ... import capability as capability_classes
from ... import ladder
from ...config import Config
from ...errors import ApiError
from ...installonsubmit import live_decisions
from ...voices import NARRATOR_ENGINE_SAMPLING
from ..context import AppContext, Routers


def installable_job_type_rows() -> list[dict[str, Any]]:
    """Every job type this BUILD knows, and what it would take to have it.

    `GET /v1/capability`'s `job_types`, PHASE13-OPERATOR.md sections 3.2a and 4.
    One row per job type named by the capability class table, in that table's
    report order, and every field read from the module that already owns it:

    | field | owner |
    |---|---|
    | `job_type`, `classes` | `crucible/capability.py`'s `CLASSES` |
    | `installer` | `crucible/cli.py`'s `INSTALLER_FOR` |
    | `narrator_engines` | `crucible/voices.py`'s `NARRATOR_ENGINE_SAMPLING` |

    `installer` is the job type `POST /v1/tasks {"type": "install"}` must be
    given to build this one's env, which is almost always itself — `denoise` is
    the exception, because it shares `rvc`'s env, and a page offering it an
    Install button of its own would be drawing a control the task door refuses
    `unknown_job_type`. `null` means nothing installs it: `echo` is compiled in.

    `narrator_engines` is empty for every type but `tts`, and for `tts` it is
    the whole of what `narrator_engine` may be — the same list the task door
    validates against, so a page cannot offer an engine the POST will refuse.
    It is a LIST and not a default: on cuda-linux the two engines cannot share
    a venv and there is no default (`crucible/tasks.py`'s
    `require_narrator_engine`).

    Whether the type is OFFERED here is deliberately absent: that is
    `/v1/setup`'s `job_types`, which is `store.registry` and the one owner of
    it. Repeating it would make a stale second answer possible in the seconds
    around an install's reload (3.4).
    """
    from ...cli import INSTALLER_FOR

    ordered: list[str] = []
    classes_of: dict[str, list[str]] = {}
    for entry in capability_classes.CLASSES:
        if entry.job_type not in classes_of:
            ordered.append(entry.job_type)
            classes_of[entry.job_type] = []
        classes_of[entry.job_type].append(entry.name)
    return [
        {
            "job_type": job_type,
            "classes": classes_of[job_type],
            "installer": INSTALLER_FOR.get(job_type),
            "narrator_engines": (
                sorted(NARRATOR_ENGINE_SAMPLING) if job_type == "tts" else []
            ),
        }
        for job_type in ordered
    ]


def register(routers: Routers, ctx: AppContext) -> None:
    private = routers.private
    config, backend = ctx.config, ctx.backend

    # ------------------------------------------------------------ capability

    @private.get("/capability/plan")
    async def capability_plan(request: Request) -> dict[str, Any]:
        """What an install or a pull will give THIS card, before it happens.

        Owen, 2026-09-26: *"that can be in a modal or something that pops up
        when the user tries to install a pakcage from the crucible ui"*. The
        operator page asks this before its Install and Pull buttons act and
        shows `confirm` in a confirmation the person accepts or cancels; it
        renders the words and writes none of them.

        `?job_type=<type>` for an install, `?subject=<id>` for a pull. Decided
        LIVE, exactly as `crucible install` will decide it: this card's size
        and generation (`backend`, `ladder.card_for`), this config's allowance
        and choices — so the modal and the record install then writes are the
        same walk. Nothing is written.
        """
        live: Config = request.app.state.config
        query = request.query_params
        job_type, subject = query.get("job_type"), query.get("subject")
        if (job_type is None) == (subject is None):
            raise ApiError(
                400,
                "invalid_request",
                "name exactly one of ?job_type= (an install) or ?subject= (a pull)",
            )
        # One walk, shared with install-on-submit, whose job status carries the
        # same sentences for a caller that has no modal (2026-09-26).
        decisions, card, pool = live_decisions(live, backend)
        if job_type is not None:
            return capability_classes.install_plan(
                job_type,
                decisions,
                card=card,
                total_bytes=backend.gpu.vram_bytes,
                pool=pool,
                desktop_allowance_bytes=live.desktop_allowance_bytes,
                desktop_basis=live.desktop_allowance_basis,
            )
        return capability_classes.subject_plan(
            subject, decisions, card=card, total_bytes=backend.gpu.vram_bytes, pool=pool
        )

    @private.get("/capability")
    async def capability(request: Request) -> dict[str, Any]:
        """What this server can hold, per capability class, and why not.

        The read a client needs before it decides what to ask for. PHASE 9 made
        the act-to-model mapping a PER-HOST fact — `crucible install` probes the
        card and picks the largest candidate that fits, so a 24 GB box serves
        `translate` with a 4-bit 27B, a bigger one serves it with something else,
        and a 12 GB box does not serve it at all. A client that was handed a model
        id by configuration would be carrying a model this server may have
        refused.

        WHY A CLASS AND NOT A JOB TYPE. `enable_llm` is one boolean and Owen
        ruled translation binary per server, so `clean` and `translate` have to be
        able to disagree. `simplify` and `analysis` became classes of their own
        too, on the naming ruling (Owen, 2026-09-13: a job is never reported
        as a different job), and `generate` is the generic chat-shaped act
        whose working context the CLIENT may state (`?class=generate&
        context_tokens=&concurrency=`, see `capability.served_rows`).

        EVERY ROW STATES ITS WORK: `work` is the working context the fit was
        computed for, with `from: "default" | "request"`, and a client-sized
        row adds `context_ceilings` — each candidate's longest servable
        request here, the smaller of the most its manifest ever starts an
        engine with on this backend (`max_context`, or `context_default` where
        none is stated) and what this host's memory affords. A size above the
        ceiling is `400 context_over_limit`, never clamped. A `load-model` with
        `params.context` is held to the same ceiling (at one in flight) and
        refused with the same body; that is how a client reaches a ceiling
        above the resident model's `max_model_len`.

        `enabled: false` IS AN ANSWER, not an error. A server that cannot
        translate says so with the number that decided it, and a client should be
        able to render "this machine cannot do that" without it looking like a
        fault.

        THIS IS A RECORD, NOT AN AUTHORITY. `[jobs] enable_*` remains the single
        owner of what this server offers; this says what the numbers were when
        somebody decided. `total_bytes` is the card the decision was made on, so a
        reader can tell a stale record from a current one — which is how a swapped
        GPU is noticed without anybody writing down a date.

        `job_types` IS NOT PART OF THE RECORD, and that is why it is added here
        rather than in `CapabilityRecord.to_dict()`. PHASE13-OPERATOR.md section
        4 draws the operator page's Job types section from this one read, and to
        draw it the page needs three things the stored record cannot carry: which
        job type each class feeds (`capability.CLASSES`), which command builds
        that type's env (`cli.INSTALLER_FOR` — `denoise` shares `rvc`'s), and
        which narrator engines a `tts` install may name
        (`voices.NARRATOR_ENGINE_SAMPLING`). All three are THIS BUILD's tables,
        read live; a record written months ago must not be able to answer them,
        because they are facts about the code, not about the card. Put in the
        record they would be a second copy that goes stale the day an engine is
        added — which is the shape R1 exists to forbid. The page holding its own
        copy is the same defect one layer out, and is what section 4 means by
        "never a hard-coded list".
        """
        live: Config = request.app.state.config
        record = live.capability
        if record is None:
            # Absent is its own answer and must not be dressed up as an empty
            # decision: a config written before `crucible capability` ran, or by a
            # build that predates it, has DECIDED NOTHING. Returning empty rows
            # would read as "probed, and nothing fit", which is the opposite news.
            raise ApiError(
                503,
                "capability_undecided",
                "this server has no capability record; nothing has probed the card "
                "on this host yet. Run `crucible capability --write` (or reinstall) "
                "to decide, and read `GET /v1/info` for what it offers meanwhile",
            )
        # EVERY ROW SAYS WHERE ITS WORK RUNS (PHASE15-HOST.md section 3.3), and
        # the answer is read off `[routes]` — the one owner of it — rather than
        # inferred from the row's `selected` carrying a slash. The two agree,
        # because `crucible/settings.py` rewrites the record from the routes on
        # every write that touches one; asking the routes is what makes them
        # unable to disagree if a record ever went stale (R1).
        # A CLIENT MAY SIZE A CLIENT-SIZED CLASS (`?class=generate&
        # context_tokens=40960&concurrency=1`), because the client is the one
        # that knows its request sizes and this server is the one that knows
        # its card (Owen, 2026-09-23: *"give it the ability to set the context
        # limit"*). Read off the raw query rather than as typed parameters so a
        # bad value is this server's named refusal (`invalid_working_context`)
        # and not the framework's 422. `capability.served_rows` owns all of it:
        # the validation, the live decision, the ceiling, and the `work` echo
        # every row now carries.
        query = request.query_params
        document = record.to_dict()
        document["classes"] = capability_classes.served_rows(
            record,
            gpu_vendor=backend.gpu.vendor,
            chosen={entry.capability: entry.model for entry in live.local_models},
            routes={entry.capability: entry.model for entry in live.routes},
            capability_class=query.get("class"),
            context_tokens=query.get(capability_classes.CONTEXT_TOKENS_PARAM),
            concurrency=query.get(capability_classes.CONCURRENCY_PARAM),
            card=ladder.card_for(config.home, backend.gpu),
        )
        for row in document["classes"]:
            row["route"] = (
                "upstream"
                if live.route_model(row["capability"]) is not None
                else "local"
            )
        return {**document, "job_types": installable_job_type_rows()}
