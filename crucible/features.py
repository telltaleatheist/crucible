"""What this server's API offers, by name, so an app checks for a feature instead of
comparing version numbers.

GET /v1/info answers `features`: the sorted names below. A name means the routes and
request fields it describes exist in this build. It does not mean the job type is
enabled on this host or that a model is installed; `job_types` and `capabilities` in
the same answer say that. A feature is added here in the same change that ships it,
and is never renamed: an app that branched on it would silently stop finding it.
"""

from __future__ import annotations

FEATURES: dict[str, str] = {
    "events": "GET /v1/events: one SSE stream of every change on the server, opening with "
    "a snapshot and resumable with Last-Event-ID (docs/EVENTS.md).",
    "events.topics": "GET /v1/events?topics=job,queue,...: only the named topics.",
    "jobs.events": "GET /v1/jobs/{id}/events: one job's own SSE stream, resumable with "
    "Last-Event-ID.",
    "jobs.hold": "`hold` on a submit and /v1/jobs/{id}/hold: a job's artifacts are kept "
    "until the client lets go of them.",
    "jobs.resume": "`params.resume` and /v1/resumable: a resumable job continues the "
    "journal an earlier run left (docs/RESUMABLE-JOBS.md).",
    "queue.jobs": "POST /v1/jobs: a busy server queues the job instead of refusing it; "
    "GET/DELETE /v1/queue and its heartbeat (docs/QUEUE.md).",
    "queue.calls": "A chat or a decision is held open in the same line until the "
    "resident model has a slot (docs/QUEUE.md).",
    "queue.default": "Every request that can wait (a job, a chat, a decision, a TTS "
    "stream) waits in the line by default; `\"queue\": false` refuses at once instead, "
    "and `{\"max_wait_s\": N}` sets the wait. `\"queue\": {}` is refused "
    "(docs/QUEUE.md).",
    "queue.events": "GET /v1/queue/events: the waiting line's own SSE stream.",
    "queue.sessions": "/v1/queue/sessions: an app's session holds the machine for a run "
    "of requests, waits its turn in the line, and ends on close or idle (docs/QUEUE.md).",
    "activity": "GET /v1/activity: what the server is doing, in one read.",
    "accelerator": "GET /v1/accelerator: what is on the card and which holders are "
    "Crucible's own.",
    "settings": "GET/PUT /v1/settings: the one door apps configure Crucible through.",
    "catalog": "GET /v1/catalog and DELETE /v1/catalog/{kind}/{id}: what this build can "
    "serve, what is on disk, and reclaiming it.",
    "tasks": "POST /v1/tasks: operator tasks (pull, install, module, engine, "
    "engine-restart) with their own SSE stream.",
    "uploads": "POST /v1/uploads: bytes too big for a request body, named by `blob_id`.",
    "playground": "GET /v1/playground: the pages the operator page's playground draws.",
    "chat": "POST /v1/openai/chat/completions: chat on the resident model, OpenAI-shaped.",
    "decide": "POST /v1/decide with `questions`: answer distributions read off the "
    "resident model's next-token logprobs.",
    "decide.items": "POST /v1/decide with `items`: one choice answer per item, in one "
    "request.",
    "models.forms": "A model may come in more than one form (precision) of the same "
    "weights; each server serves the best form its card holds, GET /v1/models lists "
    "them, and `form` on a chat, a decision or `load-model` names one "
    "(docs/FITS-AND-THE-CARD.md section 8).",
    "tts": "The `tts` job: a render of text to audio with a narration voice.",
    "tts.stream": "/v1/tts/stream: a long-lived narration session, text in and audio "
    "back over SSE.",
    "voices": "/v1/voices: the narration voices, their manifests and updates.",
    "asr": "The `asr` job: speech to text.",
    "align": "The `align` job: words placed on audio, chunk by chunk.",
    "align.longform": "The `align-longform` job: a whole book's text aligned to its "
    "audio, with cues.",
    "rvc": "The `rvc` job: voice conversion.",
    "denoise": "The `denoise` job: speech separated from what is behind it.",
    "image": "The `image` job: pictures from words (docs/IMAGE.md).",
    "image.inpaint": "`mask` on the `image` job: only the white region of a start "
    "picture is regenerated (docs/IMAGE.md).",
    "audio": "The `audio` job: sound effects, music and songs (docs/AUDIO.md).",
    "segment": "The `segment` job: subject cutouts and point-and-box selections, as "
    "masks (docs/SEGMENT.md).",
    "video": "The `video` job: clips with sound from words or a start picture "
    "(docs/VIDEO.md).",
}


def names() -> list[str]:
    return sorted(FEATURES)


__all__ = ["FEATURES", "names"]
