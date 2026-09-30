from __future__ import annotations

import sys
import urllib.parse
from dataclasses import dataclass, field
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


def activity_work(activity: Mapping[str, Any] | None) -> tuple[Progress, ...]:
    if not activity:
        return ()
    jobs = [job_progress(job) for job in activity.get("running") or []]
    jobs += [job_progress(job) for job in activity.get("queued") or []]
    return tuple(jobs + chat_progress(activity))


def home_view(status: Mapping[str, Any] | Exception, info: Any, activity: Any,
              capability: Any) -> HomeView:
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
class ActivityView:
    work: tuple[Progress, ...]
    loaded: Fact
    lease: str
    tasks: tuple[TaskLine, ...]


def activity_view(activity: Any, tasks: Any, watch: Progress | None) -> ActivityView:
    lease = (activity or {}).get("lease") if isinstance(activity, Mapping) else None
    lease_text = ""
    if lease:
        lease_text = f"{lease.get('client') or 'An app'} is keeping {lease.get('kind')} loaded for {lease.get('act')}"
    work = activity_work(activity if isinstance(activity, Mapping) else None)
    return ActivityView(
        work=((watch,) if watch is not None else ()) + work,
        loaded=resident_fact(activity if isinstance(activity, Mapping) else None),
        lease=lease_text, tasks=task_lines(tasks),
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
