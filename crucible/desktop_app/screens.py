from __future__ import annotations

import sys
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from ..platform.paths import INSTALL_ONE_LINER
from ..platform.wsl_table import RELEASE_REPOSITORY
from .api import ApiError

POSIX_INSTALL_LINE = (
    f"curl -fsSL https://github.com/{RELEASE_REPOSITORY}/releases/latest/download/install.sh | sh"
)

GIB = 1024 ** 3

JOB_TYPE_WORDS = {
    "llm": ("Language models", "Clean, translate, simplify and decide with a local LLM"),
    "tts": ("Narration", "Read text aloud in a trained voice"),
    "asr": ("Transcription", "Turn speech into text"),
    "align": ("Alignment", "Place known words at their time in the audio"),
    "rvc": ("Voice conversion", "Change one voice into another"),
    "denoise": ("Noise removal", "Clean audio and split vocals from music"),
    "image": ("Image generation", "Make pictures from a text prompt"),
    "audio": ("Audio generation", "Make sound effects, music and songs from words"),
    "segment": ("Cutouts and selections", "Cut a subject out of a picture, or select what you click"),
    "video": ("Video generation", "Make video clips with sound from words or a picture"),
}

HIDDEN_JOB_TYPES = frozenset({"echo"})

START = "start"
RESTART = "restart"
REPAIR = "repair"

OK = "ok"
WARN = "warn"
BAD = "bad"
IDLE = "idle"


def install_line(platform: str = sys.platform) -> str:
    return INSTALL_ONE_LINER if platform == "win32" else POSIX_INSTALL_LINE


def size_text(value: Any) -> str:
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    if value >= GIB:
        return f"{value / GIB:.1f} GB"
    return f"{max(1, round(value / 1024 ** 2))} MB"


def sentence(text: Any) -> str:
    words = str(text or "")
    return words[:1].upper() + words[1:]


def refusal_text(error: ApiError) -> str:
    return f"{error.message} ({error.code})"


@dataclass(frozen=True)
class Fact:
    label: str
    value: str


@dataclass(frozen=True)
class Progress:
    title: str
    detail: str
    fraction: float | None
    cancel: str | None = None
    target: str = "job"


@dataclass(frozen=True)
class HomeView:
    headline: str
    tone: str
    detail: str
    action: str | None = None
    action_label: str = ""
    facts: tuple[Fact, ...] = ()
    work: tuple[Progress, ...] = ()


def _down_action(state: str) -> tuple[str | None, str]:
    if state in ("stopped", "unreachable"):
        return START, "Start Crucible"
    if state == "broken":
        return REPAIR, "Repair Crucible"
    if state in ("unauthorized", "unhealthy", "wrong_service"):
        return RESTART, "Restart Crucible"
    return None, ""


def _down_view(status: Mapping[str, Any]) -> HomeView:
    state = str(status.get("state", "unreachable"))
    action, label = _down_action(state)
    headlines = {
        "stopped": "Crucible is stopped",
        "unreachable": "Crucible is not running",
        "broken": "Crucible needs repair",
    }
    detail = str(status.get("detail") or "")
    if action is not None:
        detail = (detail + ". " if detail else "") + f"Click {label} to fix it."
    return HomeView(headline=headlines.get(state, "Crucible is not answering"), tone=BAD,
                    detail=detail, action=action, action_label=label)


def not_installed_view(error: Exception, platform: str = sys.platform) -> HomeView:
    return HomeView(
        headline="Crucible is not set up on this computer",
        tone=BAD,
        detail=(f"{error}. Install it by pasting this into "
                f"{'PowerShell' if platform == 'win32' else 'Terminal'}:\n{install_line(platform)}"),
    )


def memory_facts(info: Mapping[str, Any], capability: Mapping[str, Any] | None) -> list[Fact]:
    host = info.get("host") or {}
    gpu = host.get("gpu") or {}
    total = (capability or {}).get("total_bytes") or gpu.get("vram_bytes")
    allowance = (capability or {}).get("desktop_allowance_bytes") or 0
    unified = gpu.get("vendor") == "apple"
    facts = []
    if gpu.get("name"):
        facts.append(Fact("Machine", str(gpu["name"])))
    if total:
        words = "unified memory" if unified else "card memory"
        kept = f", {size_text(allowance)} kept for the desktop" if allowance else ""
        facts.append(Fact("Memory", f"{size_text(total)} {words}{kept}"))
    return facts


def resident_fact(activity: Mapping[str, Any] | None) -> Fact:
    resident = (activity or {}).get("resident")
    if not resident:
        return Fact("Loaded", "Nothing is loaded; memory is free")
    words = f"{resident.get('id')} ({resident.get('kind')})"
    size = size_text(resident.get("memory_bytes_estimate"))
    held = (resident.get("held_by") or {}).get("who")
    return Fact("Loaded", words + (f", about {size}" if size else "") + (f", held by {held}" if held else ""))


def job_progress(job: Mapping[str, Any]) -> Progress:
    progress = job.get("progress")
    fraction = max(0.0, min(1.0, float(progress))) if isinstance(progress, (int, float)) else None
    who = f" for {job['client']}" if job.get("client") else ""
    model = f" with {job['model']}" if job.get("model") else ""
    detail = str(job.get("message") or job.get("status") or "")
    if job.get("status") == "queued" and job.get("position") is not None:
        detail = f"waiting, number {job['position']} in line"
    return Progress(title=f"{job.get('type')}{model}{who}", detail=detail, fraction=fraction,
                    cancel=str(job.get("job_id")) if job.get("job_id") else None)


def chat_progress(activity: Mapping[str, Any]) -> list[Progress]:
    chat = activity.get("chat") or {}
    rows = []
    for row in chat.get("rows") or []:
        who = f" for {row['client']}" if row.get("client") else ""
        rows.append(Progress(title=f"{row.get('act') or 'chat'} with {row.get('model')}{who}",
                             detail="answering", fraction=None))
    return rows


def waiting_in_queue(job: Mapping[str, Any]) -> bool:
    return job.get("waited_s") is not None


def activity_work(activity: Mapping[str, Any] | None) -> tuple[Progress, ...]:
    if not activity:
        return ()
    jobs = [job_progress(job) for job in activity.get("running") or []]
    jobs += [job_progress(job) for job in activity.get("queued") or [] if not waiting_in_queue(job)]
    return tuple(jobs + chat_progress(activity))


def waited_text(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)) or seconds < 0:
        return ""
    if seconds < 60:
        return f"{round(seconds)} s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min"


@dataclass(frozen=True)
class QueueLine:
    job_id: str
    position: int | None
    title: str
    detail: str
    waited: str
    kind: str = "job"


KIND_WORDS = {"job": "Job", "call": "Chat", "session": "Session"}


def _when(stamp: Any) -> datetime | None:
    if not isinstance(stamp, str):
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return None


def _waited_s(row: Mapping[str, Any], now: datetime | None) -> Any:
    """How long it has waited NOW: the queue document is read when the line changes,
    not every tick, so its `waited_s` is only right at the moment it was read."""
    submitted = _when(row.get("submitted"))
    if submitted is None or now is None:
        return row.get("waited_s")
    return max(0.0, (now - submitted).total_seconds())


def queue_lines(queue: Any, open_session: str | None = None,
                now: datetime | None = None) -> tuple[QueueLine, ...]:
    """The waiting line as GET /v1/queue lists it: jobs, held-open chats and decisions
    (`call`), and app sessions waiting for their turn (`session`)."""
    rows = queue.get("items") if isinstance(queue, Mapping) else None
    lines = []
    for row in rows or []:
        kind = str(row.get("kind") or "job")
        model = f" with {row['model']}" if row.get("model") else ""
        position = row.get("position")
        client = row.get("client") or "an app that did not say its name"
        what = "Session" if kind == "session" else str(row.get("type"))
        detail = f"From {client}"
        if open_session is not None and row.get("session") == open_session:
            detail += ", in the open session (it runs first)"
        lines.append(QueueLine(
            job_id=str(row.get("job_id")),
            position=position if isinstance(position, int) else None,
            title=(f"{position}. " if isinstance(position, int) else "") + f"{what}{model}",
            detail=detail,
            waited=f"waited {waited_text(_waited_s(row, now))}",
            kind=kind,
        ))
    return tuple(lines)


def queue_fact(queue: Any) -> Fact:
    rows = queue.get("items") if isinstance(queue, Mapping) else None
    if not rows:
        return Fact("Queue", "")
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind") or "job")
        counts[kind] = counts.get(kind, 0) + 1
    names = {"job": ("job", "jobs"), "call": ("chat", "chats"), "session": ("session", "sessions")}
    parts = [f"{n} {names.get(kind, (kind, kind))[0 if n == 1 else 1]}"
             for kind, n in counts.items()]
    return Fact("Queue", ", ".join(parts) + " waiting; see Activity")


def home_view(status: Mapping[str, Any] | Exception, info: Any, activity: Any,
              capability: Any, queue: Any = None) -> HomeView:
    if isinstance(status, Exception):
        return not_installed_view(status)
    if status.get("state") != "running":
        return _down_view(status)
    facts = [Fact("Version", str(status.get("version") or "")),
             Fact("Name", str(status.get("name") or "")),
             Fact("Backend", str(status.get("backend") or ""))]
    if isinstance(info, Mapping):
        facts += memory_facts(info, capability if isinstance(capability, Mapping) else None)
    if isinstance(activity, Mapping):
        facts.append(resident_fact(activity))
        facts.append(queue_fact(queue))
    work = activity_work(activity if isinstance(activity, Mapping) else None)
    detail = "Working" if work else "Ready, and nothing is running"
    return HomeView(headline="Crucible is running", tone=OK, detail=detail,
                    facts=tuple(fact for fact in facts if fact.value), work=work)


@dataclass(frozen=True)
class SubjectRow:
    kind: str
    id: str
    title: str
    subtitle: str
    size: str
    installed: bool
    resident: bool
    can_pull: bool
    can_remove: bool
    note: str = ""
    tone: str = IDLE
    extra: tuple[str, ...] = field(default_factory=tuple)


def running_task(tasks: Any) -> Mapping[str, Any] | None:
    rows = tasks.get("tasks") if isinstance(tasks, Mapping) else None
    for task in rows or []:
        if task.get("state") == "running":
            return task
    return None


def task_subject(task: Mapping[str, Any] | None) -> tuple[str, str] | None:
    if task is None or task.get("type") != "pull":
        return None
    request = task.get("request") or {}
    return str(request.get("kind")), str(request.get("id"))


def subject_note(row: Mapping[str, Any], pulling: bool) -> tuple[str, str]:
    if pulling:
        return "Downloading", WARN
    if row.get("resident"):
        return "Loaded now", OK
    if row.get("installed"):
        return "Installed", OK
    if row.get("missing_files"):
        return "Incomplete; download it again", WARN
    return "", IDLE


def subject_row(row: Mapping[str, Any], busy: Mapping[str, Any] | None) -> SubjectRow:
    installed = bool(row.get("installed"))
    resident = bool(row.get("resident"))
    size = size_text(row.get("installed_bytes") if installed else row.get("expected_bytes"))
    note, tone = subject_note(row, task_subject(busy) == (row.get("kind"), row.get("id")))
    return SubjectRow(
        kind=str(row.get("kind")), id=str(row.get("id")),
        title=str(row.get("name") or row.get("id")),
        subtitle=str(row.get("id")) if row.get("name") and row.get("name") != row.get("id") else str(row.get("job_type") or ""),
        size=size if installed or not size else f"{size} download",
        installed=installed, resident=resident,
        can_pull=not installed and busy is None,
        can_remove=installed and not resident and busy is None,
        note=note, tone=tone,
    )


def catalog_rows(catalog: Any, tasks: Any, kinds: tuple[str, ...]) -> list[SubjectRow]:
    rows = catalog.get("rows") if isinstance(catalog, Mapping) else None
    busy = running_task(tasks)
    chosen = [row for row in rows or [] if row.get("kind") in kinds]
    chosen.sort(key=lambda row: (not row.get("installed"), str(row.get("job_type")), str(row.get("id"))))
    return [subject_row(row, busy) for row in chosen]


MODEL_KINDS = ("model", "rvc", "rvc-base", "denoise", "image")


def model_rows(catalog: Any, tasks: Any) -> list[SubjectRow]:
    return catalog_rows(catalog, tasks, MODEL_KINDS)


def voice_rows(voices: Any, catalog: Any, tasks: Any, info: Any) -> list[SubjectRow]:
    by_id = {row.id: row for row in catalog_rows(catalog, tasks, ("voice",))}
    sources = (info or {}).get("voice_sources") or {} if isinstance(info, Mapping) else {}
    rows = []
    for voice in voices if isinstance(voices, list) else []:
        base = by_id.get(voice.get("id"))
        if base is None:
            continue
        label = (sources.get(voice.get("manifest")) or {}).get("label", "")
        revision = str(voice.get("revision") or "")[:8]
        pin = f"pinned to {revision}" if revision else ""
        subtitle = ", ".join(part for part in (voice.get("narrator_engine"), voice.get("language"), pin, label) if part)
        note = base.note
        if not voice.get("loadable") and voice.get("reason"):
            note = str(voice["reason"])
        rows.append(SubjectRow(
            kind="voice", id=base.id, title=str(voice.get("display") or base.id), subtitle=subtitle,
            size=base.size, installed=base.installed, resident=base.resident,
            can_pull=base.can_pull, can_remove=base.can_remove, note=note,
            tone=base.tone if voice.get("loadable") or not voice.get("reason") else WARN,
            extra=("reset",) if voice.get("manifest") == "override" else (),
        ))
    return rows


@dataclass(frozen=True)
class PackageRow:
    job_type: str
    title: str
    blurb: str
    installed: bool
    installable: bool
    engines: tuple[str, ...]
    verdict: str
    tone: str
    note: str = ""


def _verdict(classes: list[str], verdicts: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    found = [verdicts[name] for name in classes if name in verdicts]
    if not found:
        return "", IDLE
    enabled = [each for each in found if each.get("enabled")]
    if enabled:
        return sentence(enabled[0].get("summary") or enabled[0].get("reason")), OK
    return sentence(found[0].get("reason")), WARN


def package_rows(info: Any, capability: Any, tasks: Any) -> list[PackageRow]:
    if not isinstance(info, Mapping) or not isinstance(capability, Mapping):
        return []
    offered = {entry.get("job_type") for entry in info.get("capabilities") or []}
    verdicts = {row.get("capability"): row for row in capability.get("classes") or []}
    busy = running_task(tasks) is not None
    rows = []
    for entry in capability.get("job_types") or []:
        job_type = str(entry.get("job_type"))
        if job_type in HIDDEN_JOB_TYPES:
            continue
        title, blurb = JOB_TYPE_WORDS.get(job_type, (job_type, ""))
        installer = entry.get("installer")
        verdict, tone = _verdict(list(entry.get("classes") or []), verdicts)
        note = ""
        if job_type not in offered and installer not in (None, job_type):
            note = f"Comes with {JOB_TYPE_WORDS.get(str(installer), (installer,))[0]}"
        rows.append(PackageRow(
            job_type=job_type, title=title, blurb=blurb, installed=job_type in offered,
            installable=job_type not in offered and installer == job_type and not busy,
            engines=tuple(entry.get("narrator_engines") or ()), verdict=verdict, tone=tone, note=note,
        ))
    return rows


@dataclass(frozen=True)
class UpstreamRow:
    name: str
    label: str
    field: str
    configured: bool
    shown: str


def upstream_rows(settings: Any) -> list[UpstreamRow]:
    if not isinstance(settings, Mapping):
        return []
    labels = settings.get("upstream_labels") or {}
    rows = []
    for name, record in (settings.get("upstreams") or {}).items():
        field_name = "url" if "url" in record else "key"
        if field_name == "url":
            shown = str(record.get("url") or "")
        else:
            shown = f"•••• {record['key_hint']}" if record.get("key_hint") else ""
        rows.append(UpstreamRow(name=name, label=str(labels.get(name, name)), field=field_name,
                                configured=bool(record.get("configured")), shown=shown))
    return rows


def allowance_text(settings: Any) -> str:
    if not isinstance(settings, Mapping):
        return ""
    return str(settings.get("desktop_reserve") or size_text(settings.get("desktop_allowance_bytes")))


def lan_text(record: Mapping[str, Any] | None, settings: Any) -> tuple[bool, str]:
    advertised = list((settings or {}).get("lan_advertise") or []) if isinstance(settings, Mapping) else []
    if record is None:
        return False, "Only this computer can use Crucible."
    where = ", ".join(advertised) or ", ".join(record.get("authorities") or [])
    return True, f"Other computers on this network can pair with it{': ' + where if where else ''}."


@dataclass(frozen=True)
class TaskLine:
    title: str
    state: str
    tone: str
    detail: str


def task_lines(tasks: Any) -> tuple[TaskLine, ...]:
    from .progress import request_words

    rows = tasks.get("tasks") if isinstance(tasks, Mapping) else None
    tones = {"done": OK, "failed": BAD, "cancelled": WARN, "running": WARN}
    lines = []
    for task in rows or []:
        state = str(task.get("state"))
        error = task.get("error") or {}
        detail = f"{error.get('message')} ({error.get('code')})" if error.get("message") else str(task.get("message") or "")
        lines.append(TaskLine(title=request_words(task), state=state, tone=tones.get(state, IDLE), detail=detail))
    return tuple(lines)


@dataclass(frozen=True)
class SessionLine:
    """The open queue session: one app holding Crucible for a run. Not a TTS stream."""

    session_id: str
    title: str
    detail: str


def countdown_text(deadline: Any, now: datetime | None) -> str:
    when = _when(deadline)
    if when is None or now is None:
        return ""
    left = (when - now).total_seconds()
    return "any moment now" if left <= 0 else "in " + waited_text(left)


def session_line(activity: Any, now: datetime | None = None) -> SessionLine | None:
    session = activity.get("session") if isinstance(activity, Mapping) else None
    if not isinstance(session, Mapping) or not session.get("session_id"):
        return None
    client = session.get("client") or "An app"
    items = session.get("items_run") or 0
    busy = len(session.get("in_flight") or [])
    parts = [f"{items} request{'s' if items != 1 else ''} so far"]
    parts.append(f"{busy} in progress" if busy else "nothing in progress")
    stream = session.get("stream_session")
    if isinstance(stream, Mapping):
        parts.append(f"streaming narration in {stream.get('voice')}")
    idle = countdown_text(session.get("idle_deadline"), now)
    if idle:
        parts.append(f"it closes {idle} unless {client if client != 'An app' else 'the app'} "
                     "sends more")
    return SessionLine(
        session_id=str(session["session_id"]),
        title=f"{client} has Crucible to itself for {session.get('act')}",
        detail="; ".join(parts) + ". Other apps wait until it ends.",
    )


@dataclass(frozen=True)
class ActivityView:
    work: tuple[Progress, ...]
    loaded: Fact
    session: SessionLine | None
    tasks: tuple[TaskLine, ...]
    queue: tuple[QueueLine, ...] = ()


def activity_view(activity: Any, tasks: Any, watch: Progress | None, queue: Any = None,
                  now: datetime | None = None) -> ActivityView:
    work = activity_work(activity if isinstance(activity, Mapping) else None)
    session = session_line(activity, now)
    return ActivityView(
        work=((watch,) if watch is not None else ()) + work,
        loaded=resident_fact(activity if isinstance(activity, Mapping) else None),
        session=session, tasks=task_lines(tasks),
        queue=queue_lines(queue, None if session is None else session.session_id, now),
    )


@dataclass(frozen=True)
class PairingLine:
    shown: str
    line: str
    where: str


def mask_pairing(line: str) -> str:
    head, mark, _token = line.partition("#")
    return head + mark + "••••••••" if mark else line


def pairing_lines(setup: Any) -> tuple[PairingLine, ...]:
    if not isinstance(setup, Mapping):
        return ()
    lines = [line for line in setup.get("pairing") or [] if isinstance(line, str)]
    urls = [url for url in setup.get("urls") or [] if isinstance(url, str)]
    rows = []
    for index, line in enumerate(lines):
        url = urls[index] if index < len(urls) else ""
        host = urllib.parse.urlsplit(url).hostname or url
        where = "apps on this computer" if host in ("127.0.0.1", "localhost", "::1") else host
        rows.append(PairingLine(shown=mask_pairing(line), line=line, where=where))
    return tuple(rows)


@dataclass(frozen=True)
class SettingsView:
    pairing: tuple[PairingLine, ...]
    lan_supported: bool
    lan_on: bool
    lan_words: str
    upstreams: tuple[UpstreamRow, ...]
    allowance: str
    allowance_gib: str


def posix_lan_words(setup: Any) -> tuple[bool, str]:
    urls = [url for url in (setup or {}).get("urls") or []] if isinstance(setup, Mapping) else []
    shared = [url for url in urls if "127.0.0.1" not in url and "localhost" not in url]
    if shared:
        return True, "Other computers on this network can reach it at " + ", ".join(shared) + "."
    return False, ("Only this computer can use Crucible. Switching network sharing on from this "
                   "window works on Windows; on this computer it is chosen when Crucible is installed.")


def settings_view(settings: Any, setup: Any, lan_record: Mapping[str, Any] | None,
                  lan_supported: bool) -> SettingsView:
    if lan_supported:
        lan_on, words = lan_text(lan_record, settings)
    else:
        lan_on, words = posix_lan_words(setup)
    allowance = (settings or {}).get("desktop_allowance_bytes") if isinstance(settings, Mapping) else None
    return SettingsView(
        pairing=pairing_lines(setup), lan_supported=lan_supported, lan_on=lan_on, lan_words=words,
        upstreams=tuple(upstream_rows(settings)), allowance=allowance_text(settings),
        allowance_gib=f"{allowance / GIB:g}" if isinstance(allowance, (int, float)) else "",
    )
