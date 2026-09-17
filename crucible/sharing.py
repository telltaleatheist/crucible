"""Explicit, owned Tailscale TCP sharing. Never changes the engine's bind.

The host owns sharing.json and the Serve entry; tailscale_advertise in the engine
is its projection. `reconcile` repairs opted-in sharing after a network restart.
Status proves configuration and authenticated local engine access, not a remote
client's route. An existing forward is adopted only with --adopt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import API_VERSION
from .config import crucible_home
from .errors import CrucibleError
from .host.runner import ProcessRunner, Runner
from .pairing import parse_pairing_line, read_pairing_file

RECORD = "sharing.json"
TIMEOUT = 15.0


class SharingError(CrucibleError):
    pass


def _json(runner: Runner, argv: list[str]) -> dict[str, Any]:
    result = runner.run(argv, timeout_s=TIMEOUT)
    if not result.ok:
        raise SharingError(f"sharing_command_failed: {' '.join(argv)}: {result.said()}")
    try:
        data = json.loads(result.stdout)
    except ValueError as exc:
        raise SharingError(f"sharing_bad_response: {' '.join(argv)} returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise SharingError("sharing_bad_response: expected an object")
    return data


def _entry(runner: Runner, port: int) -> dict[str, Any] | None:
    data = _json(runner, ["tailscale", "serve", "status", "--json"])
    tcp = data.get("TCP", {})
    if not isinstance(tcp, dict):
        raise SharingError("sharing_bad_response: Serve TCP is not an object")
    return tcp.get(str(port))


def _matches(entry: Any, target: str) -> bool:
    return isinstance(entry, dict) and entry == {"TCPForward": target}


def _run(runner: Runner, argv: list[str]) -> None:
    result = runner.run(argv, timeout_s=TIMEOUT)
    if not result.ok:
        raise SharingError(f"sharing_command_failed: {' '.join(argv)}: {result.said()}")


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
                or not isinstance(record["target"], str)
                or not isinstance(record["authority"], str)):
            raise ValueError("invalid ownership record")
    except (ValueError, KeyError, TypeError) as exc:
        raise SharingError(f"sharing_record_invalid: {path}: {exc}") from exc
    return record


class Engine:
    """The local engine's door, shared by `crucible sharing` and `crucible lan`.

    `label` prefixes every refusal this class raises. It is REQUIRED and not
    defaulted to "sharing": a `crucible lan` failure reporting itself as
    `sharing_engine_unreachable` sends a person to read about Tailscale when
    the thing that broke was a port forward.
    """

    def __init__(self, home: Path, label: str):
        self.label = label
        line = read_pairing_file(home)
        if line is None:
            raise SharingError(f"{label}_no_pairing: register/start Crucible before sharing it")
        self.pairing = parse_pairing_line(line)
        parts = urlsplit(self.pairing.url)
        if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise SharingError(f"{label}_not_local: the local pairing file must name loopback")
        if parts.port is None:
            raise SharingError(f"{label}_missing_port: local pairing must state the engine port")
        self.target = f"127.0.0.1:{parts.port}"

    def request(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self.pairing.url + "/v1/" + path,
            data=None if body is None else json.dumps(body).encode(), method=method,
            headers={"Authorization": "Bearer " + self.pairing.token,
                     "X-Crucible-Api": str(API_VERSION),
                     "Content-Type": "application/json", "User-Agent": "crucible-sharing"},
        )
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=TIMEOUT) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise ValueError("expected an object")
            return result
        except (OSError, ValueError) as exc:
            raise SharingError(f"{self.label}_engine_unreachable: {method} /v1/{path}: {exc}") from exc

    def verify(self) -> None:
        ping = self.request("GET", "ping")
        if ping.get("crucible") is not True or ping.get("name") != self.pairing.name:
            raise SharingError(f"{self.label}_wrong_engine: local port is not the paired Crucible")
        self.request("GET", "settings")  # proves the token, not just the name

    def advertise(self, field: str, authorities: list[str]) -> None:
        """Publish this owner's addresses into ITS field, never another's."""
        result = self.request("PUT", "settings", {field: authorities})
        if result.get(field) != authorities:
            raise SharingError(
                f"{self.label}_publish_failed: engine did not confirm {field}"
            )


def enable(home: Path, runner: Runner, engine: Engine, *, port: int = 7100,
           adopt: bool = False) -> dict[str, Any]:
    if type(port) is not int or not 1 <= port <= 65535:
        raise SharingError("sharing_bad_port: expected a port from 1 to 65535")
    old = read(home)
    if old is not None and (old["port"] != port or old["target"] != engine.target):
        raise SharingError("sharing_installation_changed: disable existing sharing before changing its port or target")
    engine.verify()
    node = _json(runner, ["tailscale", "status", "--json"])
    if node.get("BackendState") != "Running":
        raise SharingError("sharing_tailscale_offline: Tailscale must be connected")
    dns = node.get("Self", {}).get("DNSName")
    if not isinstance(dns, str) or not dns.strip("."):
        raise SharingError("sharing_no_dns: Tailscale did not publish this node's DNS name")
    authority = f"{dns.rstrip('.')}:{port}"
    existing = _entry(runner, port)
    if existing is not None:
        if not _matches(existing, engine.target):
            raise SharingError("sharing_port_conflict: this Tailscale port serves something else; nothing changed")
        if old is None and not adopt:
            raise SharingError("sharing_unowned: matching forward already exists; use --adopt to transfer ownership")
    record = {"schema_version": 1, "port": port, "target": engine.target,
              "authority": authority, "state": "pending"}
    # Durable intent before a mutation: a failed publish can be reconciled.
    _write(home, record)
    if existing is None:
        _run(runner, ["tailscale", "serve", "--bg", f"--tcp={port}", "tcp://" + engine.target])
    if not _matches(_entry(runner, port), engine.target):
        raise SharingError("sharing_verification_failed: Serve did not retain the requested forward")
    engine.advertise("tailscale_advertise", [authority])
    record["state"] = "configured"
    _write(home, record)
    return {**record, "url": "http://" + authority, "remote_reachability": "not_tested"}


def disable(home: Path, runner: Runner, engine: Engine) -> dict[str, Any]:
    record = read(home)
    if record is None:
        return {"state": "disabled"}
    # Withdraw the projection first. If the engine cannot be reached, retain
    # the record and forward so a retry never loses what it needs to clean up.
    engine.advertise("tailscale_advertise", [])
    existing = _entry(runner, record["port"])
    if existing is not None:
        if not _matches(existing, record["target"]):
            raise SharingError("sharing_ownership_changed: forward was edited externally; it has been left alone")
        _run(runner, ["tailscale", "serve", f"--tcp={record['port']}", "off"])
        if _entry(runner, record["port"]) is not None:
            raise SharingError("sharing_verification_failed: Serve still has the forward")
    (home / RECORD).unlink()
    return {"state": "disabled"}


def status(home: Path, runner: Runner, engine: Engine) -> dict[str, Any]:
    record = read(home)
    if record is None:
        return {"state": "disabled", "remote_reachability": "not_tested"}
    engine.verify()
    node = _json(runner, ["tailscale", "status", "--json"])
    configured = _matches(_entry(runner, record["port"]), record["target"])
    published = engine.request("GET", "settings").get("tailscale_advertise") == [record["authority"]]
    dns = node.get("Self", {}).get("DNSName")
    address_matches = isinstance(dns, str) and f"{dns.rstrip('.')}:{record['port']}" == record["authority"]
    return {**record, "state": "configured" if configured and published and address_matches
            and node.get("BackendState") == "Running" else "degraded",
            "forward_matches": configured, "published": published,
            "remote_reachability": "not_tested"}


def reconcile(home: Path, runner: Runner | None = None) -> dict[str, Any]:
    """Repair only previously opted-in sharing, after local engine readiness."""
    record = read(home)
    if record is None:
        return {"state": "disabled"}
    runner = ProcessRunner(sys.platform, os.environ) if runner is None else runner
    return enable(home, runner, Engine(home, "sharing"), port=record["port"])


def command(args: argparse.Namespace) -> int:
    home = crucible_home()
    runner = ProcessRunner(sys.platform, os.environ)
    try:
        engine = Engine(home, "sharing")
        if args.sharing_action == "enable":
            result = enable(home, runner, engine, port=args.port, adopt=args.adopt)
        elif args.sharing_action == "reconcile":
            record = read(home)
            result = {"state": "disabled"} if record is None else enable(home, runner, engine, port=record["port"])
        elif args.sharing_action == "disable":
            result = disable(home, runner, engine)
        else:
            result = status(home, runner, engine)
        print(json.dumps(result, indent=2))
        return 1 if result.get("state") == "degraded" else 0
    except (CrucibleError, OSError, ValueError) as exc:
        print(f"crucible: {exc}", file=sys.stderr)
        return 1


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("sharing", help="manage optional Tailscale sharing without changing the bind")
    actions = parser.add_subparsers(dest="sharing_action", required=True)
    for name in ("enable", "disable", "status", "reconcile"):
        action = actions.add_parser(name)
        if name == "enable":
            action.add_argument("--port", type=int, default=7100)
            action.add_argument("--adopt", action="store_true", help="take ownership of an existing matching forward")
        action.set_defaults(func=command)
