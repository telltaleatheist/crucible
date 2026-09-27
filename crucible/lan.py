"""Explicit, owned LAN access for the WSL engine. Never changes the engine's bind.

The engine inside WSL binds `127.0.0.1` and should keep binding it. Windows
already carries `127.0.0.1` into the guest, so an app on this machine works and
always has; what no one owned was the step from *this machine's* network
addresses to that loopback. This module owns it, on exactly the terms
`crucible/sharing.py` owns the Tailscale one:

- the host owns a record (`landoor.json`) and the two Windows rows it created,
- `lan_advertise` in the engine is its PROJECTION, not its source of truth,
- `reconcile` repairs an opted-in door after the addresses change,
- an existing forward is adopted only with `--adopt`,
- and status proves CONFIGURATION plus authenticated local engine access, never
  a remote client's route. See "WHAT IS CHECKED" below for how far that goes.

WHAT IS CHECKED, AND WHAT CANNOT BE FROM HERE (FRESH-INSTALL #46, 2026-09-26)
------------------------------------------------------------------------------
kylies-pc reported `"state": "configured"` and `"remote_reachability":
"not_tested"` while its Ethernet was on the Public profile and nothing on the LAN
could reach it. Owen asked for a test from outside. What this PC can honestly
test about itself:

- **Not a dial to its own LAN address as proof of reachability.** A connection
  from this PC to its own address never leaves the machine and Windows Firewall
  does not filter it, so it answers whatever the firewall says. It IS used, for
  the one thing it proves: the forward carries a connection to the engine
  (`forward_answers`).
- **Not a dial from the WSL guest.** Its packets arrive on the vEthernet
  adapter and are judged by THAT adapter's profile, not the Ethernet's.
- **What IS read: Windows' own decision inputs, per interface** — each offered
  address's network category, the firewall's effective settings for that
  profile (on/off, "block all incoming", group policy ignoring local rules),
  and whether "Crucible engine (LAN)" is enabled for it
  (`host/landoor.admits`). That is what Windows itself decides from.
- **What stays unknowable from here:** a router that keeps its devices apart
  (guest Wi-Fi does) and third-party firewalls. The final test is the other
  computer connecting — `crucible pair <address>` there, which is also the
  step that pairs it — and the result says so in `checked`.

A Public network is never covered by widening the rule (the module docstring of
`host/landoor.py` says why). It is named, in plain words, and the person is
asked whether to mark it Private; yes adds `Set-NetConnectionProfile` to the
same single administrator prompt.

WHY THIS IS NOT `--host 0.0.0.0`
---------------------------------
Because on WSL that does nothing. The guest is NAT'd onto a private subnet
(measured on Owen's PC: guest `eth0` 192.168.227.162/20, Windows side
192.168.224.1, LAN 192.168.68.0/24), so a guest listener on `0.0.0.0` is still
invisible to the LAN. Binding wider inside the guest would widen exposure
without buying reachability — the worst of both. The crossing is a Windows-side
fact and belongs to the Windows-side host. On a native Linux or macOS server
`--host 0.0.0.0` IS the whole answer and this module has nothing to do; it
refuses to run anywhere but Windows rather than pretending otherwise.

ONE PROMPT, AND NO SCRIPT FILE
-------------------------------
Both rows go up under a single `Start-Process -Verb RunAs`. The payload is a
base64 `-EncodedCommand` rather than a temporary `.netsh`/`.cmd` file, because a
file written into a user-writable directory and then executed WITH ADMINISTRATOR
is a local privilege-escalation surface: anything that can replace it between
the write and the prompt chooses what runs as admin. An encoded command is fixed
when the argv is built, and the argv is data a test can read.

The elevated child's exit code is not the authority on success — `netsh` reports
the LAST command's status, and a person can dismiss UAC. `enable` and `disable`
both re-READ the machine afterwards and refuse on what they find, which is the
only check that cannot be fooled by either.

AND IT IS REFUSED WHEN THE ENGINE IS THE NATIVE ONE
----------------------------------------------------
PHASE19-AUTOMATIC-WSL.md 2.9. Everything above assumes the thing answering
`127.0.0.1:7100` lives inside WSL2, because that is the only shape where a
portproxy is a CROSSING. On a machine whose engine is the native Windows one
(`backend.LLAMA_WINDOWS`), the very same row forwards `0.0.0.0:7100` to
`127.0.0.1:7100` — the listener and the target are one process, and every
connection the forward accepts it makes again to itself. That is the self-loop
measured on Owen's PC on 2026-09-17, which held 15.5k of the machine's 16.4k
ephemeral ports and made localhost keepers fail at random. Under PHASE19 native
Windows stops being a transient state and becomes the OUTCOME on every machine
that cannot host WSL2, so `enable` refuses it by name.

WHO IS ASKED, AND WHY IT IS THE ENGINE ITSELF
----------------------------------------------
`GET /v1/info`'s `host.backend`, from the engine this verb is already holding
— not the orchestrator's `owner` on `:7101` and not `config.own_engine_backend`.

- The orchestrator's `owner: child` is a MEASUREMENT of what the tray spawned,
  and it is one observer away from the question: an engine it calls `found`
  (PHASE15 4.1a) is just as native and just as much a self-loop, and a tray
  that is not running has no answer at all while `:7100` still answers.
- `config.own_engine_backend` is a DECLARATION about this installation, and on
  the machine that matters it answers `None`: Owen's PC's Windows-side
  `config.toml` is an orchestrator's, with no `[server]` section, because the
  engine lives in the guest's installation.
- The engine's own document is the fact itself. It is the process listening on
  the port this row would forward to, and this module already dials it
  (`Engine.verify`), so asking it adds a field to a request that is made
  anyway rather than a second copy of "which engine answers 7100 here".

THE FIREWALL ROW IS NOT ADDED ON THE NATIVE PATH EITHER
--------------------------------------------------------
A native engine bound to a LAN address DOES need the inbound allow, so the
tempting answer is "refuse the forward, add the rule". This module does not,
for two reasons and neither is thrift:

- Nothing here knows whether that engine is bound to a LAN address. `[server]
  host` is not on `/v1/info`, and an inbound allow in front of a loopback-only
  listener admits nothing while reading, in `netsh` and in the firewall UI, as
  though the machine were open. That is the same silent no-op `landoor.py`
  refuses to ship for a Private rule on a Public-only machine.
- This module owns Windows rows THROUGH `landoor.json`, and a refusal writes no
  record. A row added beside a refusal is a row `disable` has never heard of —
  the understating record this file's own comment at the end of `enable` was
  written to prevent.

So the refusal names the rule as the thing that is still needed, and leaves the
machine exactly as it found it."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .backend import LLAMA_WINDOWS
from .config import crucible_home
from .errors import CrucibleError
from .host import landoor
from .host.paths import ENGINE_PORT
from .host.runner import ProcessRunner, RunResult, Runner
from .sharing import Engine

RECORD = "landoor.json"

#: Long enough to contain a UAC prompt a person has to notice and click. The
#: read-only probes keep `landoor`'s own short timeout; this one covers a human.
ELEVATION_TIMEOUT = 180.0

#: The dial through the forward (`forward_answers`). Short: it is this PC to
#: itself.
FORWARD_TIMEOUT = 5.0

#: How a caller hears what is happening WHILE it happens (FRESH-INSTALL #44):
#: the CLI prints to stderr; the installer and the tray pass their own line.
Say = Callable[[str], None]

#: Asked once, before the one prompt, with the Public networks that keep other
#: computers out. True marks them Private in the same prompt. None: never asked.
AskPrivate = Callable[[Sequence[landoor.NetworkInterface]], bool]

#: #44, said the moment the prompt is raised and not two minutes later.
PROMPT_WAITING = (
    "Windows is asking for administrator permission in a prompt on THIS PC's "
    "screen. Waiting up to 3 minutes for someone at this PC to click Yes."
)

#: #44, for the operator who is not at the screen. Windows' OpenSSH gives an
#: administrator's session its full rights, which `read_network` sees as
#: `elevated`, and then no prompt is raised at all.
REMOTE_NOTE = (
    "You are connected to this PC remotely, and that prompt appears only on the "
    "PC's own screen. Either someone at the PC clicks Yes, or connect with an "
    "administrator account: an administrator's remote (SSH) session already has "
    "the rights, so this then runs with no prompt."
)

#: #46: what the result can and cannot vouch for, in every `enable`/`status`.
CHECKED = (
    "Checked on this PC: the port forward reaches the engine, and Windows "
    "Firewall's own settings for each network. Not checkable from this PC: a "
    "router that keeps its devices apart (guest Wi-Fi often does) or another "
    "firewall program. The final test is the other computer connecting: "
    "`crucible pair <this PC's address>` on it, which also pairs it."
)


class LanError(CrucibleError):
    pass


def _ps_call(argv: Sequence[str]) -> str:
    """One argv as a PowerShell call operator with every word quoted."""
    program, *rest = argv
    words = " ".join("'" + word.replace("'", "''") + "'" for word in rest)
    return f"& '{program}' {words}".rstrip()


def elevated_argv(commands: Sequence[Sequence[str]]) -> list[str]:
    """Run these argvs elevated, in order, under ONE prompt.

    The inner script is base64 UTF-16LE, which is what `-EncodedCommand` takes
    and which contains only `A-Za-z0-9+/=` — so the outer PowerShell string has
    nothing in it that needs escaping, and the two levels of quoting that would
    otherwise be required cannot disagree.
    """
    script = "$ErrorActionPreference='Continue';" + ";".join(_ps_call(c) for c in commands)
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "Start-Process -Verb RunAs -Wait -FilePath 'powershell.exe' -ArgumentList "
        f"'-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand','{encoded}'",
    ]


def _write(home: Path, data: dict[str, Any]) -> None:
    home.mkdir(parents=True, exist_ok=True)
    temporary = home / (RECORD + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(home / RECORD)


def read(home: Path) -> dict[str, Any] | None:
    path = home / RECORD
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record["schema_version"] != 1 or type(record["port"]) is not int
                or not 1 <= record["port"] <= 65535
                or not isinstance(record["rule"], str)
                or not isinstance(record["authorities"], list)
                or not all(isinstance(entry, str) for entry in record["authorities"])):
            raise ValueError("invalid ownership record")
    except (ValueError, KeyError, TypeError) as exc:
        raise LanError(f"lan_record_invalid: {path}: {exc}") from exc
    return record


def _require_windows(runner: Runner) -> None:
    if runner.platform != "win32":
        raise LanError(
            "lan_not_windows: this door is the Windows-to-WSL crossing. On this "
            "platform the engine's own bind is the whole answer — set [server] "
            "host to 0.0.0.0 in config.toml, or install with --host 0.0.0.0, and "
            "open the port in whatever firewall this machine runs"
        )


def _refuse_a_native_engine(engine: Engine, port: int) -> None:
    """PHASE19 2.9 — a portproxy in front of a native engine is a self-loop.

    Reads `host.backend` off the engine's own `/v1/info`. A document that does
    not carry it is refused rather than assumed: "which engine answers this
    port" is the whole basis of the row about to be added, and an engine too
    old to say is an engine this verb cannot decide about.
    """
    info = engine.request("GET", "info")
    host = info.get("host")
    backend = host.get("backend") if isinstance(host, dict) else None
    if not isinstance(backend, str) or backend == "":
        raise LanError(
            "lan_engine_backend_unknown: the engine on 127.0.0.1:"
            f"{port} answered GET /v1/info without host.backend, so nothing "
            "here can tell whether a port forward to it would cross into WSL2 "
            "or loop back to Windows. Upgrade the engine before opening a LAN "
            "door onto it"
        )
    if backend != LLAMA_WINDOWS:
        return
    raise LanError(
        f"lan_native_engine: 127.0.0.1:{port} on this machine is answered by "
        f"the native Windows engine (backend {LLAMA_WINDOWS}), not by a Linux "
        "engine inside WSL2. This door's forward exists to cross from this "
        "machine's LAN addresses into the guest's loopback; aimed at a native "
        f"engine the same row forwards 0.0.0.0:{port} to 127.0.0.1:{port}, "
        "which is a self-loop — one of those held 15.5k of this machine's "
        "16.4k ephemeral ports on 2026-09-17 and made local connections fail "
        "at random. A native engine reaches the LAN by binding it itself: set "
        "`[server] host` in config.toml to the address it should listen on and "
        "restart the engine. An inbound allow for TCP "
        f"{port} is then the only thing Windows still needs, and this command "
        "does not add it, because nothing here can see what the engine is "
        "bound to and a rule in front of a loopback-only listener admits "
        "nothing while looking as though the machine were open"
    )


def _network(runner: Runner) -> landoor.NetworkFacts:
    try:
        return landoor.read_network(runner)
    except (ValueError, KeyError, TypeError) as exc:
        raise LanError(f"lan_addresses_unreadable: {exc}") from exc


def _candidates(facts: landoor.NetworkFacts) -> list[landoor.NetworkInterface]:
    """The addresses another machine could dial. FRESH-INSTALL #47.

    Until 2026-09-26 this was every non-loopback IPv4 address, and kylies-pc's
    pairing lines included 192.168.96.1, WSL's own vEthernet adapter, which no
    other machine can use. "Picking a subset would be a guess" was the reason
    for listing everything; it is not a guess to leave out an adapter that
    exists only inside this PC, and `landoor.offered` names exactly which.
    """
    found = landoor.offered(facts)
    if not found:
        raise LanError(
            "lan_no_addresses: this PC is not connected to any network another "
            "computer could reach it on (only its own internal adapters have "
            "addresses). Connect it to your home network, then run this again"
        )
    return found


def _networks(facts: landoor.NetworkFacts) -> list[dict[str, Any]]:
    """Per offered address: which network, and would Windows let a LAN peer in."""
    rows = []
    for interface in landoor.offered(facts):
        admitted, why = landoor.admits(interface, facts)
        rows.append({
            "address": interface.address,
            "interface": interface.alias,
            "network": interface.network,
            "category": interface.category,
            "admitted": admitted,
            "why": why,
        })
    return rows


def _forward_answers(runner: Runner, address: str, port: int) -> bool:
    """Does the forward carry a connection to the engine? See "WHAT IS CHECKED".

    `/v1/ping` needs no token. From this PC to its own address, so this proves
    the forward and the engine behind it, and nothing about the firewall.
    """
    return runner.get(f"http://{address}:{port}/v1/ping", timeout_s=FORWARD_TIMEOUT) == 200


def _remote(env: Mapping[str, str]) -> bool:
    """Is the person driving this over SSH, away from the PC's screen? #44."""
    return bool(env.get("SSH_CONNECTION") or env.get("SSH_CLIENT"))


def _verdict(networks: list[dict[str, Any]], forward: bool) -> tuple[str, str, str | None]:
    """(state, remote_reachability, the plain next step or None)."""
    blocked = [row for row in networks if not row["admitted"]]
    if not forward:
        first = networks[0]["address"] if networks else "this PC's address"
        return "degraded", "forward_not_answering", (
            f"The port forward is in place but did not reach the engine from "
            f"{first} on this PC itself. Windows' IP Helper service carries that "
            "forward; restarting this PC normally brings it back. Then run "
            "`crucible lan status`."
        )
    if not blocked:
        return "configured", "admitted_by_windows", None
    public = [row for row in blocked if row["category"] == "Public"]
    if public:
        names = ", ".join(
            f'"{row["network"] or row["interface"]}"' for row in public
        )
        return "degraded", "blocked_by_windows" if len(blocked) == len(networks) else "partly_blocked", (
            f"Other computers on {names} cannot reach Crucible: Windows has that "
            "network marked Public, which keeps them out. If it is your home or "
            "office network, run `crucible lan enable` again and answer yes when "
            "it asks to mark it Private (or add --make-private). On a cafe, hotel "
            "or other shared network, leave it Public."
        )
    return "degraded", "blocked_by_windows" if len(blocked) == len(networks) else "partly_blocked", (
        "Other computers cannot reach Crucible on "
        + "; ".join(f'{row["address"]}: {row["why"]}' for row in blocked)
        + "."
    )


def _say(say: Say | None, line: str) -> None:
    if say is not None:
        say(line)


def _apply(runner: Runner, commands: list[list[str]], *, elevated: bool,
           say: Say | None) -> RunResult | None:
    """Make the changes: straight through when this session is already an
    administrator, else under ONE prompt that is announced as it is raised.

    #44 (kylies-pc, 2026-09-26): 126 s passed with the prompt on a screen
    nobody was at, and nothing on the operator's side said so. None comes back
    for the straight-through path, where there is no prompt to diagnose.
    """
    if elevated:
        _say(say, "This session already has administrator rights, so Windows "
                  "will not show a prompt.")
        for argv in commands:
            runner.run(argv, timeout_s=ELEVATION_TIMEOUT)
        return None
    _say(say, PROMPT_WAITING)
    if _remote(runner.env):
        _say(say, REMOTE_NOTE)
    return runner.run(elevated_argv(commands), timeout_s=ELEVATION_TIMEOUT)


def _prompt_failed(result: RunResult | None, runner: Runner) -> LanError | None:
    """The prompt was not answered Yes, in the words a person needs. #44.

    None when the prompt is not what went wrong (no prompt was raised, or it
    exited 0), and the caller says what it found instead.
    """
    if result is None:
        return None
    remote = f" {REMOTE_NOTE}" if _remote(runner.env) else ""
    if result.failure is not None and "timed out" in result.failure:
        return LanError(
            "lan_admin_prompt_unanswered: nobody answered the administrator "
            "prompt on this PC's screen within 3 minutes, so nothing was "
            f"changed. Someone at this PC has to click Yes.{remote}"
        )
    if result.code not in (0, None):
        return LanError(
            "lan_admin_prompt_declined: the administrator prompt on this PC's "
            "screen was answered No (or closed), so nothing was changed. Run this "
            "again and click Yes on it"
        )
    return None


def _not_applied(result: RunResult | None, runner: Runner, detail: str) -> LanError:
    """Why `enable`'s rows are not there."""
    return _prompt_failed(result, runner) or LanError(
        "lan_verification_failed: the rows are not both there after asking "
        f"for them ({detail}). If the administrator prompt was "
        "dismissed, nothing was changed; run this again and accept it"
    )


def _ask_on_terminal(public: Sequence[landoor.NetworkInterface]) -> bool:
    """The CLI's `AskPrivate`: one plain question on stderr, one line of stdin.

    Stdout stays the JSON document. Anything but yes is no.
    """
    names = ", ".join(interface.label for interface in public)
    plural = len(public) > 1
    print(
        f"\nThis PC's network{'s' if plural else ''} {names} "
        f"{'are' if plural else 'is'} marked Public, so Windows keeps the other "
        "computers on it out.\n"
        "If it is your home or office network, Crucible can mark it Private so "
        "they can use this PC.\n"
        "Say no on a cafe, hotel, airport or any other shared network.\n"
        f"Mark {names} as Private? [y/N] ",
        end="", file=sys.stderr, flush=True,
    )
    answer = sys.stdin.readline()
    return answer.strip().lower() in ("y", "yes")


def enable(home: Path, runner: Runner, engine: Engine, *, port: int = ENGINE_PORT,
           adopt: bool = False, ask_private: AskPrivate | None = None,
           say: Say | None = None) -> dict[str, Any]:
    if type(port) is not int or not 1 <= port <= 65535:
        raise LanError("lan_bad_port: expected a port from 1 to 65535")
    _require_windows(runner)
    old = read(home)
    if old is not None and old["port"] != port:
        raise LanError(
            "lan_installation_changed: disable the existing LAN door before "
            "changing its port"
        )
    engine.verify()
    # BEFORE `detect`, and long before anything is written or elevated: on a
    # native machine there is no door to open, so there is nothing to read the
    # machine for either.
    _refuse_a_native_engine(engine, port)
    door = landoor.detect(runner, port)
    if door.mechanism == landoor.MIRRORED:
        raise LanError(
            "lan_mirrored: this machine's .wslconfig asks for mirrored "
            "networking, where the guest is already on this host's interfaces "
            "and a forward would be a second, redundant hop. Nothing to add"
        )
    if door.forward and old is None and not adopt:
        raise LanError(
            "lan_unowned: a port forward for this port already exists and "
            "Crucible did not create it; use --adopt to take ownership of it"
        )
    facts = _network(runner)
    candidates = _candidates(facts)
    # #46: a Public network with nothing admitting TCP on it is named and, if
    # the person says it is theirs, marked Private in the SAME prompt as the
    # rows. Asked BEFORE anything is written: a question is not a mutation.
    shut = [
        interface for interface in candidates
        if interface.profile == "Public" and not landoor.admits(interface, facts)[0]
    ]
    to_mark = list(shut) if shut and ask_private is not None and ask_private(shut) else []
    record: dict[str, Any] = {
        "schema_version": 1,
        "port": port,
        "rule": landoor.RULE_NAME,
        # Every candidate, for now: a half-finished enable overstates.
        "authorities": [f"{interface.address}:{port}" for interface in candidates],
        "state": "pending",
    }
    if to_mark:
        # Recorded so `status` can say which network Crucible changed. It is
        # NOT reverted by `disable`: the person said the network is theirs,
        # and that stays true when Crucible stops sharing on it.
        record["made_private"] = [interface.label for interface in to_mark]
    # Durable intent before a mutation, exactly as `sharing.enable` does it: a
    # publish that dies half way leaves a record `reconcile` can finish from.
    _write(home, record)
    missing = [
        command
        for present, command in (
            (door.forward, landoor.add_argv(port)),
            (door.firewall, landoor.firewall_add_argv(port)),
        )
        if not present
    ] + [landoor.make_private_argv(interface) for interface in to_mark]
    result = _apply(runner, missing, elevated=facts.elevated, say=say) if missing else None
    after = landoor.detect(runner, port)
    if not (after.forward and after.firewall):
        raise _not_applied(result, runner, after.detail)
    # THE ROWS EXIST NOW, AND THE RECORD SAYS SO BEFORE THE PUBLISH IS TRIED.
    #
    # This is the durable-intent write completed, not repeated: the first one
    # said "about to change the machine", this one says "the machine IS
    # changed". Without it a failing `advertise` left the file reading
    # `pending` -- "nothing has been opened yet" -- while the port was open and
    # verified, which is a record lying in the QUIET direction. Measured on
    # Owen's PC 2026-09-17 and reported by the Foundry session: `lan enable`
    # took its netsh rows, then failed publishing to a 0.6.12 engine that does
    # not know `lan_advertise`, and the file understated the exposure from then
    # on. A half-finished enable must overstate what it did, never understate:
    # the record is what `disable` uses to know what there is to clean up.
    record["state"] = "open"
    _write(home, record)
    # Read again: the prompt may have changed a network's category, and what
    # is published is only what Windows now admits (#46, #47).
    networks = _networks(_network(runner))
    authorities = [f"{row['address']}:{port}" for row in networks if row["admitted"]]
    record["authorities"] = authorities
    record["private_network"] = bool(authorities)
    _write(home, record)
    engine.advertise("lan_advertise", authorities)
    forward = _forward_answers(runner, candidates[0].address, port)
    state, reachability, following = _verdict(networks, forward)
    record["state"] = state
    _write(home, record)
    report: dict[str, Any] = {
        **record,
        "urls": [f"http://{authority}" for authority in authorities],
        "detail": after.detail,
        "networks": networks,
        "forward_answers": forward,
        "remote_reachability": reachability,
        "checked": CHECKED,
    }
    if following is not None:
        report["next"] = following
    return report


def disable(home: Path, runner: Runner, engine: Engine, *,
            say: Say | None = None) -> dict[str, Any]:
    record = read(home)
    if record is None:
        return {"state": "disabled"}
    _require_windows(runner)
    # Withdraw the projection FIRST. If the engine cannot be reached, the record
    # and the rows are retained so a retry still knows what it has to clean up.
    engine.advertise("lan_advertise", [])
    port = record["port"]
    try:
        elevated = _network(runner).elevated
    except LanError:
        # Only decides prompt-or-not; not knowing means the prompt, which is
        # the path that works for everyone.
        elevated = False
    result = _apply(
        runner, [landoor.remove_argv(port), landoor.firewall_remove_argv(port)],
        elevated=elevated, say=say,
    )
    after = landoor.detect(runner, port)
    if after.forward or after.firewall:
        refused = _prompt_failed(result, runner)
        if refused is not None:
            # #44: the prompt was not answered Yes; say that, not "still has".
            # The record is kept either way, so a retry knows what to remove.
            raise refused
        raise LanError(
            "lan_verification_failed: Windows still has "
            + " and ".join(
                name for name, present in
                (("the port forward", after.forward), (f'"{record["rule"]}"', after.firewall))
                if present
            )
            + "; the record has been kept so this can be retried"
        )
    (home / RECORD).unlink()
    return {"state": "disabled"}


def status(home: Path, runner: Runner, engine: Engine) -> dict[str, Any]:
    record = read(home)
    if record is None:
        return {"state": "disabled", "remote_reachability": "not_tested"}
    engine.verify()
    door = landoor.detect(runner, record["port"])
    published = engine.request("GET", "settings").get("lan_advertise")
    facts = _network(runner)
    networks = _networks(facts)
    current = [f"{row['address']}:{record['port']}" for row in networks if row["admitted"]]
    addresses_match = published == record["authorities"] == current
    forward = bool(networks) and _forward_answers(runner, networks[0]["address"], record["port"])
    state, reachability, following = _verdict(networks, forward)
    configured = door.forward and door.firewall and state == "configured"
    if not (door.forward and door.firewall) and following is None:
        following = (
            "Windows no longer has both of Crucible's rows (" + door.detail + "). "
            "Run `crucible lan enable` to put them back."
        )
    elif not addresses_match and following is None:
        following = (
            "This PC's addresses changed since sharing was turned on, so the "
            "addresses other computers were given are out of date. Run "
            "`crucible lan reconcile` to update them."
        )
    report: dict[str, Any] = {
        **record,
        "state": "configured" if configured and addresses_match else "degraded",
        "forward": door.forward,
        "firewall": door.firewall,
        "private_network": bool(current),
        "published": published,
        "addresses_match": addresses_match,
        "detail": door.detail,
        "networks": networks,
        "forward_answers": forward,
        "remote_reachability": reachability,
        "checked": CHECKED,
    }
    if following is not None:
        report["next"] = following
    return report


def reconcile(home: Path, runner: Runner | None = None, *,
              engine: Engine | None = None) -> dict[str, Any]:
    """Repair only a previously opted-in door.

    The addresses are the reason this exists: a DHCP lease changes and the
    published `lan_advertise` becomes a list of places nothing answers. `enable`
    recomputes them from the OS and republishes, and it adds no Windows row that
    is already there, so this is cheap and prompts for nothing on the ordinary
    path where only the addresses moved.
    """
    record = read(home)
    if record is None:
        return {"state": "disabled"}
    runner = ProcessRunner(sys.platform, os.environ) if runner is None else runner
    engine = Engine(home, "lan") if engine is None else engine
    return enable(home, runner, engine, port=record["port"], adopt=True)


def _to_stderr(line: str) -> None:
    print(f"crucible: {line}", file=sys.stderr, flush=True)


def command(args: argparse.Namespace) -> int:
    home = crucible_home()
    runner = ProcessRunner(sys.platform, os.environ)
    try:
        if args.lan_action == "explain":
            print(landoor.ELEVATION_SENTENCE)
            return 0
        engine = Engine(home, "lan")
        if args.lan_action == "enable":
            # #46: --make-private answers the Public-network question up front;
            # at a terminal it is asked; with neither, nobody is asked and a
            # Public network is reported in `next`, never changed.
            if args.make_private:
                ask: AskPrivate | None = lambda _public: True
            elif sys.stdin is not None and sys.stdin.isatty():
                ask = _ask_on_terminal
            else:
                ask = None
            result = enable(home, runner, engine, port=args.port, adopt=args.adopt,
                            ask_private=ask, say=_to_stderr)
        elif args.lan_action == "reconcile":
            result = reconcile(home, runner)
        elif args.lan_action == "disable":
            result = disable(home, runner, engine, say=_to_stderr)
        else:
            result = status(home, runner, engine)
        print(json.dumps(result, indent=2))
        if result.get("next"):
            # The JSON is for a script; this line is for the person (#46).
            print(f"crucible: {result['next']}", file=sys.stderr)
        return 1 if result.get("state") == "degraded" else 0
    except (CrucibleError, OSError, ValueError) as exc:
        print(f"crucible: {exc}", file=sys.stderr)
        return 1


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "lan",
        help="let other devices on this network reach the WSL engine (Windows)",
    )
    actions = parser.add_subparsers(dest="lan_action", required=True)
    for name in ("enable", "disable", "status", "reconcile", "explain"):
        action = actions.add_parser(name)
        if name == "enable":
            action.add_argument("--port", type=int, default=ENGINE_PORT)
            action.add_argument(
                "--adopt",
                action="store_true",
                help="take ownership of a port forward that already exists",
            )
            action.add_argument(
                "--make-private",
                action="store_true",
                help="if this PC's network is marked Public, mark it Private "
                     "without asking (only for your own home or office network)",
            )
        action.set_defaults(func=command)
