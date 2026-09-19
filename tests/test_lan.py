"""The LAN door as an operator verb.

The mechanism (`crucible/host/landoor.py`) is pinned in `test_host.py`. This
file is about the VERB: what it refuses, what it writes down before it changes
anything, which field it publishes into, and what it does when the machine does
not end up in the state that was asked for.

The fake runner flips its own answers when the elevated argv goes past, which is
how a test on Linux exercises a Windows mutation: the point is never that `netsh`
ran, it is that the verb re-read the machine afterwards and believed THAT. It
RECORDS argv and executes nothing, so nothing in this file touches this
machine's own portproxy table.
"""
from __future__ import annotations

import base64
import json
import re

import pytest

from crucible import lan
from crucible.errors import CrucibleError
from crucible.host import landoor
from crucible.host.runner import RunResult

WINDOWS_ENV = {"USERPROFILE": r"C:\Users\tellt"}

#: The two addresses `ipv4_addresses` is made to report here: a LAN one and a
#: tailnet one. Both get their own row — see `_addresses` in `crucible/lan.py`.
LAN_ADDRESS = "192.168.68.100"
TAILNET_ADDRESS = "100.64.0.1"
BOTH = [LAN_ADDRESS, TAILNET_ADDRESS]

PRIVATE = json.dumps([{"InterfaceAlias": "Ethernet 2", "NetworkCategory": "Private"}])
PUBLIC = json.dumps([{"InterfaceAlias": "Wi-Fi", "NetworkCategory": "Public"}])


def listing(forwards: list[str], port: int = 7100) -> str:
    """`netsh interface portproxy show v4tov4`, as this machine would print it."""
    if not forwards:
        return ""
    return (
        "Listen on ipv4:             Connect to ipv4:\n\n"
        "Address         Port        Address         Port\n"
        "--------------- ----------  --------------- ----------\n"
    ) + "".join(
        f"{address:<16}{port:<12}{'127.0.0.1':<16}{port}\n" for address in forwards
    )


def ok(stdout: str = "") -> RunResult:
    return RunResult(code=0, stdout=stdout, stderr="", failure=None)


def absent() -> RunResult:
    """How `netsh` reports a rule that is not there: non-zero, localised text."""
    return RunResult(code=1, stdout="\nNo rules match the specified criteria.\n",
                     stderr="", failure=None)


class Runner:
    """Answers the four probes `landoor.detect` makes, and can be mutated.

    The portproxy table is a LIST of listen addresses rather than a boolean,
    because the wildcard row that has to come out and the per-address rows that
    go in are different rows on the same machine, and a boolean cannot tell
    them apart.
    """

    platform = "win32"

    def __init__(self, *, forwards: list[str] | None = None, firewall: bool = False,
                 profile: str = PRIVATE, applies: bool = True,
                 removes: bool = True) -> None:
        self.forwards: list[str] = [] if forwards is None else list(forwards)
        self.firewall = firewall
        self.profile = profile
        #: False models a dismissed UAC prompt: the argv runs, nothing changes.
        self.applies = applies
        #: False models the OTHER `netsh` answer, measured: a delete replies
        #: "requires elevation", exits 0, and leaves the row where it was. Adds
        #: still land, so only the removals are refused.
        self.removes = removes
        self.calls: list[list[str]] = []
        self.env = dict(WINDOWS_ENV)

    @property
    def forward(self) -> bool:
        return bool(self.forwards)

    def _apply(self, script: str) -> None:
        """Do to the fake table what the elevated script says, row by row."""
        for statement in script.split(";"):
            words = re.findall(r"'([^']*)'", statement)
            if "portproxy" in words:
                listen = [word.split("=", 1)[1] for word in words
                          if word.startswith("listenaddress=")][0]
                if "add" in words:
                    if listen not in self.forwards:
                        self.forwards.append(listen)
                elif self.removes and listen in self.forwards:
                    self.forwards.remove(listen)
            elif "advfirewall" in words:
                if "add" in words:
                    self.firewall = True
                elif self.removes:
                    self.firewall = False

    def run(self, argv, *, timeout_s, env=None) -> RunResult:
        self.calls.append(list(argv))
        joined = " ".join(argv)
        if "type" in joined and ".wslconfig" in joined:
            return ok("[wsl2]\nmemory=13GB\n")
        if "portproxy show" in joined:
            return ok(listing(self.forwards))
        if "advfirewall firewall show" in joined:
            return ok("Rule Name: Crucible engine (LAN)\n") if self.firewall else absent()
        if "Get-NetConnectionProfile" in joined:
            return ok(self.profile)
        if "Start-Process" in joined:
            if self.applies:
                self._apply(base64.b64decode(argv[-1].split("'")[-2]).decode("utf-16-le"))
            return ok()
        raise AssertionError(f"unscripted call: {joined}")

    @property
    def elevations(self) -> list[list[str]]:
        return [c for c in self.calls if "Start-Process" in " ".join(c)]


class Engine:
    """The local engine door. Records which FIELD was published into."""

    target = "127.0.0.1:7100"

    def __init__(self) -> None:
        self.published: dict[str, list[str]] = {}
        self.verified = 0

    def verify(self) -> None:
        self.verified += 1

    def advertise(self, field: str, authorities: list[str]) -> None:
        self.published[field] = authorities

    def request(self, *args):
        return {"lan_advertise": self.published.get("lan_advertise", [])}


@pytest.fixture(autouse=True)
def _addresses(monkeypatch):
    monkeypatch.setattr(lan, "ipv4_addresses", lambda: list(BOTH))


# ------------------------------------------------------------- the payload


def _decoded(argv: list[str]) -> str:
    return base64.b64decode(argv[-1].split("'")[-2]).decode("utf-16-le")


def test_one_prompt_carries_both_rows_and_names_no_temporary_file() -> None:
    argv = lan.elevated_argv(
        [landoor.add_argv(LAN_ADDRESS, 7100), landoor.firewall_add_argv(7100)]
    )
    assert argv[0] == "powershell.exe"
    assert sum("Start-Process" in word for word in argv) == 1, "exactly one prompt"
    script = _decoded(argv)
    assert "interface' 'portproxy' 'add'" in script
    assert "advfirewall' 'firewall' 'add'" in script
    assert landoor.RULE_NAME in script
    # A script FILE would be a user-writable thing executed as administrator.
    assert ".cmd" not in script and ".netsh" not in script and ".ps1" not in script


def test_the_encoded_payload_is_quote_safe_by_construction() -> None:
    """Two levels of quoting that cannot disagree.

    OUTER: base64 is `A-Za-z0-9+/=`, so the PowerShell string that carries the
    payload has nothing in it needing escape, whatever the argv contained.

    INNER: a single quote inside an argument is DOUBLED, which is how
    PowerShell spells a literal one inside a single-quoted string. The decoded
    script therefore holds the ESCAPED form and PowerShell parses it back to
    the original — asserting the raw form appeared would be asserting that the
    escaping had not happened.
    """
    argv = lan.elevated_argv([["netsh", "x", "name=it's got 'quotes'"]])
    blob = argv[-1].split("'")[-2]
    assert set(blob) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )
    assert "'name=it''s got ''quotes'''" in _decoded(argv)


# ------------------------------------------------------------- refusals


def test_this_door_refuses_to_exist_anywhere_but_windows(tmp_path) -> None:
    runner = Runner()
    runner.platform = "linux"
    with pytest.raises(lan.LanError, match="lan_not_windows"):
        lan.enable(tmp_path, runner, Engine())
    assert runner.calls == [], "it did not probe a machine it cannot change"


def test_mirrored_networking_is_refused_as_nothing_to_add(tmp_path, monkeypatch) -> None:
    runner = Runner()
    monkeypatch.setattr(
        runner, "run",
        lambda argv, *, timeout_s, env=None: ok("networkingMode=mirrored\n")
        if "wslconfig" in " ".join(argv) else ok(""),
    )
    with pytest.raises(lan.LanError, match="lan_mirrored"):
        lan.enable(tmp_path, runner, Engine())


def test_a_forward_we_did_not_create_needs_adopt(tmp_path) -> None:
    runner = Runner(forwards=["0.0.0.0"])
    with pytest.raises(lan.LanError, match="lan_unowned"):
        lan.enable(tmp_path, runner, Engine())
    assert not (tmp_path / lan.RECORD).exists(), "a refusal claims nothing"
    assert runner.elevations == [], "a refusal changes nothing"
    assert lan.enable(tmp_path, runner, Engine(), adopt=True)["state"] == "configured"


def test_a_machine_with_no_address_publishes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lan, "ipv4_addresses", lambda: [])
    with pytest.raises(lan.LanError, match="lan_no_addresses"):
        lan.enable(tmp_path, Runner(), Engine())


# ------------------------------------------------------------- the happy path


def test_enable_opens_both_rows_publishes_and_records(tmp_path) -> None:
    runner, engine = Runner(), Engine()
    result = lan.enable(tmp_path, runner, engine)

    assert result["state"] == "configured"
    assert result["remote_reachability"] == "not_tested", "nothing here dialled in"
    assert result["urls"] == ["http://192.168.68.100:7100", "http://100.64.0.1:7100"]

    # Published into ITS OWN field. Writing the tailnet's would silently drop a
    # tailnet address the moment either door was disabled.
    assert engine.published == {
        "lan_advertise": ["192.168.68.100:7100", "100.64.0.1:7100"]
    }
    assert "tailscale_advertise" not in engine.published

    record = lan.read(tmp_path)
    assert record["state"] == "configured"
    assert record["rule"] == landoor.RULE_NAME
    assert record["port"] == 7100
    assert len(runner.elevations) == 1, "both rows, one prompt"


def test_a_row_that_is_already_there_is_not_asked_for_again(tmp_path) -> None:
    runner = Runner(forwards=list(BOTH), firewall=True)
    lan.enable(tmp_path, runner, Engine(), adopt=True)
    assert runner.elevations == [], "nothing was missing, so nobody was prompted"


def test_only_the_missing_row_is_requested(tmp_path) -> None:
    runner = Runner(forwards=list(BOTH), firewall=False)
    lan.enable(tmp_path, runner, Engine(), adopt=True)
    script = _decoded(runner.elevations[0])
    assert "advfirewall" in script
    assert "portproxy" not in script, "the forward was already there"


# ------------------------------------------------------------- the sad paths


def test_a_dismissed_prompt_is_a_refusal_and_leaves_the_record_pending(tmp_path) -> None:
    runner = Runner(applies=False)
    engine = Engine()
    with pytest.raises(lan.LanError, match="lan_verification_failed"):
        lan.enable(tmp_path, runner, engine)
    # The verb re-read the machine and did not believe the exit code.
    assert engine.published == {}, "nothing is advertised that is not reachable"
    assert lan.read(tmp_path)["state"] == "pending", "durable intent survives"


def test_a_forward_with_no_rule_is_not_called_open(tmp_path) -> None:
    """The half-open state this whole module exists to stop being silent."""
    runner = Runner(forwards=list(BOTH), firewall=False, applies=False)
    with pytest.raises(lan.LanError, match="lan_verification_failed"):
        lan.enable(tmp_path, runner, Engine(), adopt=True)


def test_rows_on_a_public_only_network_are_reported_degraded(tmp_path) -> None:
    runner, engine = Runner(profile=PUBLIC), Engine()
    result = lan.enable(tmp_path, runner, engine)
    assert result["state"] == "degraded"
    assert "admits nothing here" in result["detail"]
    # Still published: the rows ARE there and the person may fix the network
    # category. What is refused is calling it open.
    assert engine.published["lan_advertise"] != []


# ------------------------------------------------------------- disable


def test_disable_withdraws_the_projection_before_it_removes_the_rows(tmp_path) -> None:
    runner, engine = Runner(), Engine()
    lan.enable(tmp_path, runner, engine)
    order: list[str] = []
    engine.advertise = lambda field, authorities: order.append(f"advertise:{authorities}")
    real_run = runner.run

    def watched(argv, *, timeout_s, env=None):
        if "Start-Process" in " ".join(argv):
            order.append("netsh")
        return real_run(argv, timeout_s=timeout_s, env=env)

    runner.run = watched
    assert lan.disable(tmp_path, runner, engine)["state"] == "disabled"
    assert order == ["advertise:[]", "netsh"], "withdrawn first, then removed"
    assert not (tmp_path / lan.RECORD).exists()
    assert runner.forward is False and runner.firewall is False


def test_disable_that_does_not_remove_keeps_the_record_to_retry(tmp_path) -> None:
    runner, engine = Runner(), Engine()
    lan.enable(tmp_path, runner, engine)
    runner.applies = False
    with pytest.raises(lan.LanError, match="lan_verification_failed"):
        lan.disable(tmp_path, runner, engine)
    assert (tmp_path / lan.RECORD).exists(), "still ours to clean up"


def test_disabling_what_was_never_enabled_says_so_and_touches_nothing(tmp_path) -> None:
    runner = Runner()
    assert lan.disable(tmp_path, runner, Engine()) == {"state": "disabled"}
    assert runner.calls == []


# ------------------------------------------------------------- status


def test_status_is_degraded_when_the_address_moved(tmp_path, monkeypatch) -> None:
    runner, engine = Runner(), Engine()
    lan.enable(tmp_path, runner, engine)
    assert lan.status(tmp_path, runner, engine)["state"] == "configured"
    # A DHCP lease changes. The published list now names somewhere nothing is.
    monkeypatch.setattr(lan, "ipv4_addresses", lambda: ["192.168.68.207"])
    report = lan.status(tmp_path, runner, engine)
    assert report["state"] == "degraded"
    assert report["addresses_match"] is False


def test_reconcile_repoints_the_rows_when_the_lease_moves(tmp_path, monkeypatch) -> None:
    """It costs one prompt now, and that is the price of naming the addresses.

    A wildcard row needed no repair when a lease moved because it listened on
    everything — including `127.0.0.1`, which is what made it dial itself. Rows
    that name an address have to be re-pointed, so `reconcile` takes the stale
    ones out and puts the new one in under a single elevation.
    """
    runner, engine = Runner(), Engine()
    lan.enable(tmp_path, runner, engine)
    monkeypatch.setattr(lan, "ipv4_addresses", lambda: ["192.168.68.207"])
    before = len(runner.elevations)
    result = lan.reconcile(tmp_path, runner, engine=engine)
    assert result["state"] == "configured"
    assert engine.published["lan_advertise"] == ["192.168.68.207:7100"]
    assert runner.forwards == ["192.168.68.207"], "the old rows went with the lease"
    assert len(runner.elevations) == before + 1, "one prompt, both the delete and the add"


def test_status_will_not_call_a_machine_that_dials_itself_configured(tmp_path) -> None:
    """Both rows, a Private network, matching addresses — and still degraded.

    A wildcard row adopted before this rule existed passes every other test
    `status` makes, so a report that re-derived "configured" from those tests
    alone called the port-eating machine healthy.
    """
    runner, engine = Runner(forwards=["0.0.0.0"], firewall=True), Engine()
    lan.enable(tmp_path, runner, engine, adopt=True)
    assert lan.status(tmp_path, runner, engine)["state"] == "configured"
    # Somebody puts the old row back by hand, next to the correct ones.
    runner.forwards.insert(0, "0.0.0.0")
    report = lan.status(tmp_path, runner, engine)
    assert report["state"] == "degraded"
    assert report["addresses_match"] is True, "the published list is not what is wrong"
    assert report["forward"] is True and report["firewall"] is True
    assert "forwards to itself" in report["detail"]


def test_reconcile_repairs_nothing_that_was_never_opted_into(tmp_path) -> None:
    assert lan.reconcile(tmp_path, Runner(), engine=Engine()) == {"state": "disabled"}


def test_a_corrupt_record_is_a_refusal_not_an_empty_door(tmp_path) -> None:
    (tmp_path / lan.RECORD).write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(CrucibleError, match="lan_record_invalid"):
        lan.read(tmp_path)


def test_a_failed_publish_records_that_the_port_IS_open(tmp_path) -> None:
    """A half-finished enable must overstate what it did, never understate.

    The rows go in, the publish fails (an engine too old to know
    `lan_advertise` refuses it with a 400 — exactly what happened on Owen's PC
    on 2026-09-17). The file must not still read `pending`, because `pending`
    means "nothing has been opened yet" and the port is open. The record is also
    what `disable` reads to know what there is to clean up.
    """
    class Refusing(Engine):
        def advertise(self, field, authorities):
            raise lan.LanError("lan_publish_failed: this engine does not know that field")

    runner = Runner()
    with pytest.raises(lan.LanError, match="lan_publish_failed"):
        lan.enable(tmp_path, runner, Refusing())

    record = lan.read(tmp_path)
    assert record["state"] == "open", "the rows are there and the record says so"
    assert record["state"] != "pending", "understating an open port is the defect"
    assert record["authorities"], "it names what it opened"
    # And the machine really is open, so the record is not merely optimistic.
    assert runner.forward is True and runner.firewall is True


def test_disable_cleans_up_after_a_failed_publish(tmp_path) -> None:
    """The record left by a half-finished enable is still enough to shut it."""
    class Refusing(Engine):
        def advertise(self, field, authorities):
            if getattr(self, "refuse", True):
                raise lan.LanError("lan_publish_failed: nope")

    runner, engine = Runner(), Refusing()
    with pytest.raises(lan.LanError):
        lan.enable(tmp_path, runner, engine)
    engine.refuse = False
    assert lan.disable(tmp_path, runner, engine)["state"] == "disabled"
    assert runner.forward is False and runner.firewall is False
    assert not (tmp_path / lan.RECORD).exists()


# ------------------------------------------------- the wildcard self-loop


def test_no_argv_this_verb_composes_ever_listens_on_the_wildcard(tmp_path) -> None:
    """`0.0.0.0` is a listen set that CONTAINS the address the row forwards TO.

    Measured on Owen's PC 2026-09-17: the portproxy service accepted its own
    connection on 127.0.0.1:7100 and dialled itself, and 15.5k of the machine's
    16.4k ephemeral ports ended in TIME_WAIT with localhost keepers failing at
    random. Every argv the verb would hand to `netsh` is read here, not only the
    one a happy path produces.
    """
    runner = Runner()
    lan.enable(tmp_path, runner, Engine())
    for call in runner.calls:
        text = _decoded(call) if "Start-Process" in " ".join(call) else " ".join(call)
        assert "0.0.0.0" not in text, f"a wildcard listener was composed: {text}"


def test_a_listen_address_that_covers_its_own_target_is_refused_by_name() -> None:
    """One rule, and it is about the LISTEN SET, not about which engine is behind it.

    `llama-windows` puts the engine itself on 127.0.0.1:7100 and the WSL engine
    is reached through the same loopback address, so naming the backend would
    have been two rules for one fact. This asks the only question that matters:
    does what the row would answer on include what it would dial?
    """
    with pytest.raises(CrucibleError, match="portproxy_self_loop"):
        landoor.add_argv("0.0.0.0", 7100)
    with pytest.raises(CrucibleError, match="portproxy_self_loop"):
        landoor.add_argv(landoor.CONNECT_ADDRESS, 7100)
    assert landoor.covers_connect_address("0.0.0.0") is True
    assert landoor.covers_connect_address(LAN_ADDRESS) is False
    # The REMOVAL is not refused: a wildcard row that already exists is exactly
    # the thing this has to be able to take back out.
    assert "listenaddress=0.0.0.0" in landoor.remove_argv("0.0.0.0", 7100)


def test_one_row_per_address_this_machine_actually_has(tmp_path) -> None:
    runner = Runner()
    lan.enable(tmp_path, runner, Engine())
    script = _decoded(runner.elevations[0])
    assert f"'listenaddress={LAN_ADDRESS}'" in script
    assert f"'listenaddress={TAILNET_ADDRESS}'" in script
    assert runner.forwards == BOTH, "one row each, and nothing wider"


def test_reconcile_takes_out_a_wildcard_row_it_finds_and_repoints_the_rest(
        tmp_path, monkeypatch) -> None:
    """Owen's PC today: one `0.0.0.0` row, added before this rule existed."""
    runner, engine = Runner(forwards=["0.0.0.0"], firewall=True), Engine()
    lan.enable(tmp_path, runner, engine, adopt=True)
    assert runner.forwards == BOTH, "the self-loop row is gone and named rows replaced it"
    monkeypatch.setattr(lan, "ipv4_addresses", lambda: ["192.168.68.207"])
    assert lan.reconcile(tmp_path, runner, engine=engine)["state"] == "configured"
    assert runner.forwards == ["192.168.68.207"]


def test_a_delete_that_needs_elevation_and_exits_zero_is_a_named_refusal(tmp_path) -> None:
    """`netsh` answers "requires elevation" and still exits 0 — measured.

    So the exit code is no more the authority on a removal than it is on an
    addition. The listing is re-READ, and a row that survived is named rather
    than reported as a door that is now correct.
    """
    runner, engine = Runner(forwards=["0.0.0.0"], firewall=True, removes=False), Engine()
    with pytest.raises(lan.LanError, match="lan_rows_not_removed"):
        lan.enable(tmp_path, runner, engine, adopt=True)
    assert "0.0.0.0" in runner.forwards, "it is still there, which is why this refused"
    assert engine.published == {}, "nothing is advertised while the loop is still running"
