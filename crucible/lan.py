from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .atomicjson import write_json
from .backend import LLAMA_WINDOWS
from .config import crucible_home
from .errors import CrucibleError
from .platform import lan_door
from .platform.paths import ENGINE_PORT
from .platform.powershell import POWERSHELL, quote, runas_argv
from .platform.runner import ProcessRunner, RunResult, Runner
from .sharing import PairedEngine

RECORD = "landoor.json"

ELEVATION_TIMEOUT = 180.0

FORWARD_TIMEOUT = 5.0

Say = Callable[[str], None]

AskPrivate = Callable[[Sequence[lan_door.NetworkInterface]], bool]

PROMPT_WAITING = (
    "Windows is asking for administrator permission in a prompt on THIS PC's "
    "screen. Waiting up to 3 minutes for someone at this PC to click Yes."
)

REMOTE_NOTE = (
    "You are connected to this PC remotely, and that prompt appears only on the "
    "PC's own screen. Either someone at the PC clicks Yes, or connect with an "
    "administrator account: an administrator's remote (SSH) session already has "
    "the rights, so this then runs with no prompt."
)

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
    program, *rest = argv
    words = " ".join(quote(word) for word in rest)
    return f"& {quote(program)} {words}".rstrip()


def elevated_argv(commands: Sequence[Sequence[str]]) -> list[str]:
    script = "$ErrorActionPreference='Continue';" + ";".join(_ps_call(c) for c in commands)
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return runas_argv([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded])


def _write(home: Path, data: dict[str, Any]) -> None:
    write_json(home / RECORD, data)


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


def _refuse_a_native_engine(engine: PairedEngine, port: int) -> None:
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


def _network(runner: Runner) -> lan_door.NetworkFacts:
    try:
        return lan_door.read_network(runner)
    except (ValueError, KeyError, TypeError) as exc:
        raise LanError(f"lan_addresses_unreadable: {exc}") from exc


def _candidates(facts: lan_door.NetworkFacts) -> list[lan_door.NetworkInterface]:
    found = lan_door.offered(facts)
    if not found:
        raise LanError(
            "lan_no_addresses: this PC is not connected to any network another "
            "computer could reach it on (only its own internal adapters have "
            "addresses). Connect it to your home network, then run this again"
        )
    return found


def _networks(facts: lan_door.NetworkFacts) -> list[dict[str, Any]]:
    rows = []
    for interface in lan_door.offered(facts):
        admitted, why = lan_door.admits(interface, facts)
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
    return runner.get(f"http://{address}:{port}/v1/ping", timeout_s=FORWARD_TIMEOUT) == 200


def _remote(env: Mapping[str, str]) -> bool:
    return bool(env.get("SSH_CONNECTION") or env.get("SSH_CLIENT"))


def _verdict(networks: list[dict[str, Any]], forward: bool) -> tuple[str, str, str | None]:
    blocked = [row for row in networks if not row["admitted"]]
    if not forward:
        first = networks[0]["address"] if networks else "this PC's address"
        return "degraded", "forward_not_answering", (
            f"The port forward is in place but did not reach the engine from "
            f"{first} on this PC itself. Windows' IP Helper service carries that "
            "forward, and a restart normally brings it back: open Start, click "
            'the power button and choose "Update and restart" (or "Restart" if '
            "that is all there is), and sign in again afterwards. Then run "
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
    return _prompt_failed(result, runner) or LanError(
        "lan_verification_failed: Windows does not have both the port forward "
        f"and the firewall rule Crucible asked for ({detail}). If the "
        "administrator prompt was closed, nothing was changed; run this again "
        "and click Yes on it"
    )


def _ask_on_terminal(public: Sequence[lan_door.NetworkInterface]) -> bool:
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


def _door_to_enable(home: Path, runner: Runner, engine: PairedEngine, port: int,
                    adopt: bool) -> lan_door.LanDoor:
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
    _refuse_a_native_engine(engine, port)
    door = lan_door.detect(runner, port)
    if door.mechanism == lan_door.MIRRORED:
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
    return door


def _networks_to_mark(candidates: list[lan_door.NetworkInterface],
                      facts: lan_door.NetworkFacts,
                      ask_private: AskPrivate | None) -> list[lan_door.NetworkInterface]:
    shut = [
        interface for interface in candidates
        if interface.profile == "Public" and not lan_door.admits(interface, facts)[0]
    ]
    return list(shut) if shut and ask_private is not None and ask_private(shut) else []


def _pending_record(port: int, candidates: list[lan_door.NetworkInterface],
                    to_mark: list[lan_door.NetworkInterface]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "port": port,
        "rule": lan_door.RULE_NAME,
        "authorities": [f"{interface.address}:{port}" for interface in candidates],
        "state": "pending",
    }
    if to_mark:
        record["made_private"] = [interface.label for interface in to_mark]
    return record


def _missing_commands(door: lan_door.LanDoor, port: int,
                      to_mark: list[lan_door.NetworkInterface]) -> list[list[str]]:
    return [
        command
        for present, command in (
            (door.forward, lan_door.add_argv(port)),
            (door.firewall, lan_door.firewall_add_argv(port)),
        )
        if not present
    ] + [lan_door.make_private_argv(interface) for interface in to_mark]


def _publish_admitted(home: Path, runner: Runner, engine: PairedEngine,
                      record: dict[str, Any], port: int) -> list[dict[str, Any]]:
    networks = _networks(_network(runner))
    authorities = [f"{row['address']}:{port}" for row in networks if row["admitted"]]
    record["authorities"] = authorities
    record["private_network"] = bool(authorities)
    _write(home, record)
    engine.advertise("lan_advertise", authorities)
    return networks


def enable(home: Path, runner: Runner, engine: PairedEngine, *, port: int = ENGINE_PORT,
           adopt: bool = False, ask_private: AskPrivate | None = None,
           say: Say | None = None) -> dict[str, Any]:
    door = _door_to_enable(home, runner, engine, port, adopt)
    facts = _network(runner)
    candidates = _candidates(facts)
    to_mark = _networks_to_mark(candidates, facts, ask_private)
    record = _pending_record(port, candidates, to_mark)
    _write(home, record)
    missing = _missing_commands(door, port, to_mark)
    result = _apply(runner, missing, elevated=facts.elevated, say=say) if missing else None
    after = lan_door.detect(runner, port)
    if not (after.forward and after.firewall):
        raise _not_applied(result, runner, after.detail)
    record["state"] = "open"
    _write(home, record)
    networks = _publish_admitted(home, runner, engine, record, port)
    forward = _forward_answers(runner, candidates[0].address, port)
    state, reachability, following = _verdict(networks, forward)
    record["state"] = state
    _write(home, record)
    report: dict[str, Any] = {
        **record,
        "urls": [f"http://{authority}" for authority in record["authorities"]],
        "detail": after.detail,
        "networks": networks,
        "forward_answers": forward,
        "remote_reachability": reachability,
        "checked": CHECKED,
    }
    if following is not None:
        report["next"] = following
    return report


def disable(home: Path, runner: Runner, engine: PairedEngine, *,
            say: Say | None = None) -> dict[str, Any]:
    record = read(home)
    if record is None:
        return {"state": "disabled"}
    _require_windows(runner)
    engine.advertise("lan_advertise", [])
    port = record["port"]
    try:
        elevated = _network(runner).elevated
    except LanError:
        elevated = False
    result = _apply(
        runner, [lan_door.remove_argv(port), lan_door.firewall_remove_argv(port)],
        elevated=elevated, say=say,
    )
    after = lan_door.detect(runner, port)
    if after.forward or after.firewall:
        refused = _prompt_failed(result, runner)
        if refused is not None:
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


def status(home: Path, runner: Runner, engine: PairedEngine) -> dict[str, Any]:
    record = read(home)
    if record is None:
        return {"state": "disabled", "remote_reachability": "not_tested"}
    engine.verify()
    door = lan_door.detect(runner, record["port"])
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
              engine: PairedEngine | None = None) -> dict[str, Any]:
    record = read(home)
    if record is None:
        return {"state": "disabled"}
    runner = ProcessRunner(sys.platform, os.environ) if runner is None else runner
    engine = PairedEngine(home, "lan") if engine is None else engine
    return enable(home, runner, engine, port=record["port"], adopt=True)


def _to_stderr(line: str) -> None:
    print(f"crucible: {line}", file=sys.stderr, flush=True)


def _asker(args: argparse.Namespace) -> AskPrivate | None:
    if args.make_private:
        return lambda _public: True
    if sys.stdin is not None and sys.stdin.isatty():
        return _ask_on_terminal
    return None


def _answer(args: argparse.Namespace, home: Path, runner: Runner) -> dict[str, Any]:
    engine = PairedEngine(home, "lan")
    if args.lan_action == "enable":
        return enable(home, runner, engine, port=args.port, adopt=args.adopt,
                      ask_private=_asker(args), say=_to_stderr)
    if args.lan_action == "reconcile":
        return reconcile(home, runner)
    if args.lan_action == "disable":
        return disable(home, runner, engine, say=_to_stderr)
    return status(home, runner, engine)


def command(args: argparse.Namespace) -> int:
    home = crucible_home()
    runner = ProcessRunner(sys.platform, os.environ)
    try:
        if args.lan_action == "explain":
            print(lan_door.ELEVATION_SENTENCE)
            return 0
        result = _answer(args, home, runner)
        print(json.dumps(result, indent=2))
        if result.get("next"):
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
