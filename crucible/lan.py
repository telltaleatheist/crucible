"""Explicit, owned LAN access for the WSL engine. Never changes the engine's bind.

The engine inside WSL binds `127.0.0.1` and should keep binding it. Windows
already carries `127.0.0.1` into the guest, so an app on this machine works and
always has; what no one owned was the step from *this machine's* network
addresses to that loopback. This module owns it, on exactly the terms
`crucible/sharing.py` owns the Tailscale one:

- the host owns a record (`landoor.json`) and the Windows rows it created,
- `lan_advertise` in the engine is its PROJECTION, not its source of truth,
- `reconcile` repairs an opted-in door after the addresses change,
- an existing forward is adopted only with `--adopt`,
- and status proves CONFIGURATION plus authenticated local engine access, never
  a remote client's route — nothing here has tested that a friend's laptop can
  actually reach this machine, and it says `not_tested` rather than implying it.

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

ONE ROW PER ADDRESS, NEVER THE WILDCARD
----------------------------------------
Until 2026-09-18 this added a single row listening on `0.0.0.0`. That is not
"all the LAN addresses": it is every address this machine answers on,
`127.0.0.1` included — which is the address the row forwards TO. The portproxy
service therefore accepted its own connection and dialled itself. Measured on
Owen's PC 2026-09-17: 15.5k of 16.4k ephemeral ports in TIME_WAIT, with
localhost keepers failing at random.

So the addresses are ENUMERATED and each one gets its own row, and
`landoor.add_argv` refuses `portproxy_self_loop` before any argv is composed.
Naming the addresses is also what lets `reconcile` see that one has moved,
which a wildcard row never could.

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
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import crucible_home
from .errors import CrucibleError
from .host import landoor
from .host.paths import ENGINE_PORT
from .host.runner import ProcessRunner, Runner
from .interfaces import InterfaceError, ipv4_addresses
from .sharing import Engine

RECORD = "landoor.json"

#: Long enough to contain a UAC prompt a person has to notice and click. The
#: read-only probes keep `landoor`'s own short timeout; this one covers a human.
ELEVATION_TIMEOUT = 180.0


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


def addresses() -> list[str]:
    """Every address this door may listen on, asked of the OS.

    EVERY one of them, and one row each. Picking a subset would mean deciding
    which of a person's networks is "the" LAN, which is a guess this codebase
    does not get to make; the tailnet address gets a row for the same reason
    `pairing.reachable_urls` already lists it.

    `ipv4_addresses` excludes loopback, link-local and the unspecified address,
    which is why `door_commands` needs no second predicate to keep `0.0.0.0`
    out of the listen set: an address this function returns is by construction
    an address a row may listen on.
    """
    try:
        found = ipv4_addresses()
    except InterfaceError as exc:
        raise LanError(f"lan_addresses_unreadable: {exc}") from exc
    if not found:
        raise LanError(
            "lan_no_addresses: this machine has no non-loopback IPv4 address, so "
            "opening a forward would publish nothing an app could dial"
        )
    return found


def _authorities(found: Sequence[str], port: int) -> list[str]:
    """Those same addresses as `host:port`, which is what gets published."""
    return [f"{address}:{port}" for address in found]


def door_commands(door: landoor.LanDoor, found: Sequence[str],
                  port: int) -> list[list[str]]:
    """The rows this machine is missing, and the rows it has to lose.

    The removals are ONE rule: any row listening somewhere `found` does not
    name comes out. Because `addresses()` never reports `0.0.0.0` or a loopback
    address, that single rule takes out both the stale row a DHCP move left
    behind and the wildcard self-loop row Owen's PC carries today — no second
    predicate that could drift out of step with
    `landoor.covers_connect_address`.

    Removals are ordered FIRST so that a machine whose wildcard row is being
    replaced spends no moment carrying both it and the row that supersedes it.
    """
    commands = [
        landoor.remove_argv(listen, port)
        for listen in door.forwards
        if listen not in found
    ]
    commands += [
        landoor.add_argv(address, port)
        for address in found
        if address not in door.forwards
    ]
    if not door.firewall:
        commands.append(landoor.firewall_add_argv(port))
    return commands


def _require_rows(after: landoor.LanDoor, found: Sequence[str], port: int) -> None:
    """The machine as it is NOW, believed over any exit code.

    `netsh` answers a delete it may not perform with "requires elevation" and
    still exits 0, so a row is gone only when the listing no longer shows it.
    Both directions are checked because both are silent otherwise: a row that
    would not go is a self-loop still running, and a row that would not come is
    a door reported open onto nothing.
    """
    remaining = [listen for listen in after.forwards if listen not in found]
    if remaining:
        raise LanError(
            "lan_rows_not_removed: Windows still forwards port "
            f"{port} from {', '.join(remaining)}, which is not an address this "
            "machine has"
            + ("" if not after.self_loops else
               f" — and {', '.join(after.self_loops)} covers "
               f"{landoor.CONNECT_ADDRESS}, so that row dials itself and eats "
               "this machine's ephemeral ports")
            + '. `netsh` answers a delete it may not perform with "requires '
            'elevation" and still exits 0, so run this again and accept the '
            "administrator prompt"
        )
    missing = [address for address in found if address not in after.forwards]
    if missing or not after.firewall:
        raise LanError(
            "lan_verification_failed: the rows are not all there after asking "
            f"for them ({after.detail}). If the administrator prompt was "
            "dismissed, nothing was changed; run this again and accept it"
        )


def enable(home: Path, runner: Runner, engine: Engine, *, port: int = ENGINE_PORT,
           adopt: bool = False) -> dict[str, Any]:
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
    listens = addresses()
    authorities = _authorities(listens, port)
    record: dict[str, Any] = {
        "schema_version": 1,
        "port": port,
        "rule": landoor.RULE_NAME,
        "authorities": authorities,
        "state": "pending",
    }
    # Durable intent before a mutation, exactly as `sharing.enable` does it: a
    # publish that dies half way leaves a record `reconcile` can finish from.
    _write(home, record)
    commands = door_commands(door, listens, port)
    if commands:
        runner.run(elevated_argv(commands), timeout_s=ELEVATION_TIMEOUT)
    after = landoor.detect(runner, port)
    _require_rows(after, listens, port)
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
    record["private_network"] = after.private_network
    _write(home, record)
    engine.advertise("lan_advertise", authorities)
    record["state"] = "configured" if after.private_network is not False else "degraded"
    _write(home, record)
    return {
        **record,
        "urls": [f"http://{authority}" for authority in authorities],
        "detail": after.detail,
        "remote_reachability": "not_tested",
    }


def disable(home: Path, runner: Runner, engine: Engine) -> dict[str, Any]:
    record = read(home)
    if record is None:
        return {"state": "disabled"}
    _require_windows(runner)
    # Withdraw the projection FIRST. If the engine cannot be reached, the record
    # and the rows are retained so a retry still knows what it has to clean up.
    engine.advertise("lan_advertise", [])
    port = record["port"]
    # Read the rows off the MACHINE rather than off the record's addresses: a
    # lease that moved between the last reconcile and now would otherwise leave
    # a row nothing owns, and a wildcard row adopted from before this rule
    # existed is not in the record's address list at all.
    door = landoor.detect(runner, port)
    runner.run(
        elevated_argv(
            [landoor.remove_argv(listen, port) for listen in door.forwards]
            + [landoor.firewall_remove_argv(port)]
        ),
        timeout_s=ELEVATION_TIMEOUT,
    )
    after = landoor.detect(runner, port)
    if after.forward or after.firewall:
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
    current = _authorities(addresses(), record["port"])
    addresses_match = published == record["authorities"] == current
    # `door.open` rather than the three facts re-tested here, which is what
    # this line used to do. `detect` already folds in the fourth: a machine
    # carrying a row whose listen set covers the connect address has both rows,
    # a Private network and matching addresses, and is still not a door anyone
    # wants — it is the port-eating self-loop. Re-deriving "configured" here
    # meant one of the two readers could forget a term, and this one had.
    configured = door.open
    return {
        **record,
        "state": "configured" if configured and addresses_match else "degraded",
        "forward": door.forward,
        "firewall": door.firewall,
        "private_network": door.private_network,
        "published": published,
        "addresses_match": addresses_match,
        "detail": door.detail,
        "remote_reachability": "not_tested",
    }


def reconcile(home: Path, runner: Runner | None = None, *,
              engine: Engine | None = None) -> dict[str, Any]:
    """Repair only a previously opted-in door.

    The addresses are the reason this exists: a DHCP lease changes and the
    published `lan_advertise` becomes a list of places nothing answers. `enable`
    recomputes them from the OS, re-points the rows and republishes.

    It asks for administrator when the rows actually have to move, which is the
    price of naming the addresses rather than listening on `0.0.0.0`. The
    wildcard row needed no repair because it listened on everything — including
    the loopback address it forwarded to, which is what made it dial itself. A
    reconcile that changes nothing still prompts for nothing.
    """
    record = read(home)
    if record is None:
        return {"state": "disabled"}
    runner = ProcessRunner(sys.platform, os.environ) if runner is None else runner
    engine = Engine(home, "lan") if engine is None else engine
    return enable(home, runner, engine, port=record["port"], adopt=True)


def command(args: argparse.Namespace) -> int:
    home = crucible_home()
    runner = ProcessRunner(sys.platform, os.environ)
    try:
        if args.lan_action == "explain":
            print(landoor.ELEVATION_SENTENCE)
            return 0
        engine = Engine(home, "lan")
        if args.lan_action == "enable":
            result = enable(home, runner, engine, port=args.port, adopt=args.adopt)
        elif args.lan_action == "reconcile":
            result = reconcile(home, runner)
        elif args.lan_action == "disable":
            result = disable(home, runner, engine)
        else:
            result = status(home, runner, engine)
        print(json.dumps(result, indent=2))
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
        action.set_defaults(func=command)
