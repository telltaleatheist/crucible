"""`crucible host` — the wiring. PHASE15-HOST.md section 4.

Everything this file does is decided somewhere else: `presence.py` decides what
state the machine is in, `menu.py` decides what the menu says, `installer.py`
decides what an install is, `door.py` decides who may ask for one. This is the
loop that holds them, and it is deliberately dull.

THE SHAPE, IN ORDER
--------------------
1. One host per machine (`host_already_running`): a second tray would boot the
   same distro twice and watch the first one's recoveries.
2. The Startup item, written if absent — 4.1 makes these verbs the ONE owner of
   that file and `install.ps1` calls the verb rather than writing a `.lnk`.
3. Presence: distro present → boot the guest; absent → start the
   `llama-windows` server as a child (section 0's amendment: Windows IS a
   backend, and the machine has an engine either way).
4. The pairing file, so an app on this machine never asks anybody to type a
   token.
5. The door, on loopback.
6. The tray, and a watch every 15 s.
"""

from __future__ import annotations

import os
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import VERSION
from . import installer, menu, startup
from .catalog import CatalogPort, GuestCatalog, HttpCatalog
from .door import InstallDoor, serve
from .errors import HostError
from .log import HostLog
from .menu import Distro, Engine
from .paths import (
    CONSOLE_CMD,
    ENGINE_PORT,
    console_cmd_path,
    crucible_root,
    engine_url,
    host_pack_dir,
    log_path,
    previous_log_path,
)
from .presence import Presence, PresenceWatcher
from .runner import ProcessRunner, Runner
from .wsl_states import CRUCIBLE_DISTRO

#: A second tray is refused by a file, not by a mutex: the file NAMES the
#: process that holds it, so "another one is running" is a sentence with a pid
#: in it rather than a silent exit.
LOCK_NAME = "host.pid"

#: The release the guest install fetches. The host ships AT the server's
#: version — the one legitimate default in here, for `install.ts`'s reason:
#: the bootstrapper ships at the server's version, so the default IS the
#: answer rather than a guess at one.
DEFAULT_RELEASE = VERSION

INSTALL_SH_URL = (
    "https://github.com/telltaleatheist/crucible/releases/download/"
    "v{release}/install.sh"
)


@dataclass
class HostContext:
    """Everything the loop holds. One object, so a test can build one."""

    runner: Runner
    log: HostLog
    home: Path
    watcher: PresenceWatcher
    presence: Presence
    release: str = DEFAULT_RELEASE


def read_token(home: Path) -> str | None:
    """The engine token out of the host's own config, or None before there is one.

    Parsed with the stdlib's TOML reader and not with a regex: a token is a
    quoted string and `tomllib` is what `crucible/config.py` reads the same
    file with, so the two cannot disagree about escaping.
    """
    import tomllib

    path = Path(home) / "config.toml"
    if not path.is_file():
        return None
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    auth = document.get("auth")
    if not isinstance(auth, dict):
        return None
    token = auth.get("token")
    return token if isinstance(token, str) and token else None


def acquire(home: Path) -> Path:
    """One host per machine. Refuses `host_already_running`, naming the pid."""
    home.mkdir(parents=True, exist_ok=True)
    lock = home / LOCK_NAME
    if lock.is_file():
        previous = lock.read_text(encoding="utf-8").strip()
        if previous.isdigit() and _alive(int(previous)):
            raise HostError(
                "host_already_running",
                f"another `crucible host` is running on this machine (pid {previous}). "
                "Two trays would boot the same distro twice and watch each other's "
                "recoveries. Quit that one from its menu, or end that process.",
            )
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def _alive(pid: int) -> bool:
    """Is this pid a live process? A stale lock must not wedge the tray."""
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def server_argv(env: "os._Environ[str] | dict[str, str]") -> list[str]:
    """`crucible serve` as the host's CHILD — the `llama-windows` server.

    Through the pack's own `crucible.cmd`, which is the relocatable entry point
    (4.4): `Scripts\\crucible.exe` bakes the build tree's interpreter path into
    the binary and does not survive the move to `%LOCALAPPDATA%`.
    """
    return [str(console_cmd_path(env)), "serve"]


def init_argv(env: "os._Environ[str] | dict[str, str]") -> list[str]:
    """The first-run `crucible init` for the host-mode server.

    `--backend llama-windows` and NOT `none`: section 0's amendment. This
    machine runs an engine, it just runs llama.cpp rather than vLLM.
    """
    return [str(console_cmd_path(env)), "init", "--backend", "llama-windows"]


def open_console(home: Path, log: HostLog) -> None:
    """4.2's Open console: the pairing line's URL with `#token=`.

    The DEFAULT browser, deliberately. PHASE13 5.3's hardened-window rule is
    about not putting a token into something that will navigate away with it;
    a browser the person chose is the one place a URL belongs.
    """
    from ..pairing import read_pairing_file

    line = read_pairing_file(home)
    if line is None:
        log.write(
            f"open console: there is no pairing file in {home}, so there is no "
            "server on this machine to open yet"
        )
        return
    # `crucible://name@host:port/#token` -> `http://host:port/#token=...`
    from urllib.parse import urlsplit

    parts = urlsplit(line)
    url = f"http://{parts.netloc.rsplit('@', 1)[-1]}/#token={parts.fragment}"
    log.write(f"open console: {parts.netloc.rsplit('@', 1)[-1]}")
    webbrowser.open(url)


def open_log(log: HostLog) -> None:
    """4.2's Open log. The default handler for a `.log`, which is Notepad."""
    if sys.platform == "win32":
        os.startfile(str(log.path))  # type: ignore[attr-defined]  # noqa: S606
        return
    log.write(f"the log is {log.path}")


class Host:
    """The loop. Started by `crucible host`, stopped by its Quit item."""

    def __init__(self, context: HostContext) -> None:
        self._c = context
        self._icon: object | None = None
        self._stop = threading.Event()
        self._door_server: object | None = None

    # -------------------------------------------------------------- presence

    def start(self) -> Presence:
        """Boot whichever server this machine runs, and say which."""
        distro, detail = self._c.watcher.probe_distro()
        self._c.log.write(f"presence: {detail}")
        if distro is Distro.PRESENT:
            self._c.presence = self._c.watcher.boot()
        else:
            self._c.presence = self._start_host_mode(distro)
        self._c.log.write(
            f"presence: {self._c.presence.distro.value}/"
            f"{self._c.presence.engine.value} — {self._c.presence.detail}"
        )
        return self._c.presence

    def _start_host_mode(self, distro: Distro) -> Presence:
        """The `llama-windows` server, as this process's child."""
        env = dict(self._c.runner.env)
        cmd = console_cmd_path(env)
        if not Path(str(cmd)).is_file():
            return Presence(
                distro,
                Engine.FAILED,
                f"there is no {CONSOLE_CMD} in {host_pack_dir(env)}, so this host "
                "has no server to start. Reinstall with install.ps1.",
            )
        config = self._c.home / "config.toml"
        if not config.is_file():
            first = self._c.runner.run(init_argv(env), timeout_s=300.0)
            self._c.log.write(
                f"init: {'ok' if first.ok else first.said()}"
            )
            if not first.ok:
                return Presence(distro, Engine.FAILED, f"crucible init: {first.said()}")
        self._c.watcher.respawn_host_mode(server_argv(env), env)
        if self._c.watcher._wait_for_ping(30.0):  # noqa: SLF001 - one object, one loop
            return Presence(distro, Engine.RUNNING, "the Windows engine answered /v1/ping")
        return Presence(
            distro,
            Engine.FAILED,
            "the Windows engine did not answer within 30 s — open the log",
        )

    # ------------------------------------------------------------------ menu

    def model(self) -> menu.MenuModel:
        return menu.menu_model(self._c.presence.distro, self._c.presence.engine)

    def on_click(self, item_id: str) -> None:
        self._c.log.write(f"menu: {item_id}")
        if item_id == menu.OPEN_CONSOLE:
            open_console(self._c.home, self._c.log)
        elif item_id == menu.INSTALL_ENGINE:
            # 4.7: the tray does NOT run the sequence. The switch is a control
            # on the page, the page posts the task, the server relays to this
            # host's door. One door, one sequence, one place a person watches.
            open_console(self._c.home, self._c.log)
        elif item_id == menu.RESTART_ENGINE:
            self._c.presence = self._c.watcher.boot()
        elif item_id == menu.STOP_ENGINE:
            self._stop_engine()
        elif item_id == menu.OPEN_LOG:
            open_log(self._c.log)
        elif item_id == menu.QUIT:
            self.quit()
        self._refresh()

    def _stop_engine(self) -> None:
        if self._c.presence.distro is Distro.PRESENT:
            # `systemctl --user stop`, which `Restart=always` respects: the
            # unit is STOPPED, not exited (the ruling in crucible/service.py).
            result = self._c.runner.run(
                [
                    "wsl.exe",
                    "-d",
                    self._c.watcher._distro,  # noqa: SLF001
                    "--exec",
                    "systemctl",
                    "--user",
                    "stop",
                    "crucible",
                ],
                timeout_s=60.0,
            )
            self._c.log.write(f"stop: {'ok' if result.ok else result.said()}")
        else:
            self._c.watcher.stop_child()
        self._c.presence = Presence(
            self._c.presence.distro, Engine.STOPPED, "stopped from the menu"
        )

    def _refresh(self) -> None:
        if self._icon is None:
            return
        from . import tray

        tray.update(self._icon, self.model(), self.on_click)

    # ----------------------------------------------------------------- watch

    def watch(self) -> None:
        """4.1's watch. One recovery per down-edge, then a state with a name."""
        while not self._stop.wait(self._c.watcher.watch_s):
            before = self._c.presence.engine
            self._c.presence = self._c.watcher.poll(self._c.presence.distro)
            if self._c.presence.engine is not before:
                self._c.log.write(
                    f"watch: {before.value} -> {self._c.presence.engine.value} — "
                    f"{self._c.presence.detail}"
                )
                self._refresh()

    def quit(self) -> None:
        self._stop.set()
        if self._c.presence.distro is not Distro.PRESENT:
            # In host mode the server is this process's child and 4.2's Quit
            # label already said it goes too.
            self._c.watcher.stop_child()
        if self._icon is not None:
            self._icon.stop()  # type: ignore[attr-defined]


def run(argv: list[str] | None = None) -> int:
    """`crucible host`. Returns an exit code; never raises past here."""
    env = os.environ
    home = Path(str(crucible_root(env)))
    log = HostLog(Path(str(log_path(env))), Path(str(previous_log_path(env))))
    runner = ProcessRunner(sys.platform, env)
    log.write(f"crucible host {VERSION} starting; CRUCIBLE_HOME={home}")
    acquire(home)

    # The Startup item, if absent. 4.1 makes these verbs its ONE owner, and a
    # host that has been started by hand is a host that should still be there
    # after the next login.
    try:
        outcome = startup.install(runner)
        log.write(f"startup: {outcome.detail}")
    except HostError as exc:
        log.write(f"startup: NOT written — {exc.code}: {exc.message}")

    watcher = PresenceWatcher(runner, log)
    context = HostContext(
        runner=runner,
        log=log,
        home=home,
        watcher=watcher,
        presence=Presence(Distro.UNKNOWN, Engine.STARTING, "starting"),
    )
    host = Host(context)
    host.start()
    _write_pairing(context)

    door = InstallDoor(
        log,
        _sequence(context),
        token=lambda: read_token(context.home),
    )
    try:
        serve(door)
        log.write("door: listening on 127.0.0.1:7101")
    except OSError as exc:
        log.write(f"door: NOT listening ({exc}); the page's engine switch will refuse")

    threading.Thread(target=host.watch, name="crucible-watch", daemon=True).start()

    from . import tray

    icon = tray.make_icon(host.model(), host.on_click)
    host._icon = icon  # noqa: SLF001 - one object, one loop
    icon.run()
    return 0


def _sequence(context: HostContext) -> Callable[[Callable[[installer.Event], None]], None]:
    """Bind the install sequence to THIS host's two servers.

    The catalogs are built per RUN and not once at startup, because the token
    they both use is the one the config has at the moment the move begins —
    and `migrate-config`, a few steps earlier in that same run, is what makes
    the guest's token the Windows one. A port that captured a token at tray
    start would be a port holding a token the guest never had.

    `None` for either side is a fact rather than a fallback: a machine with no
    Windows config has no Windows engine, so there is no catalog to move from
    and `migrate-weights` says exactly that.
    """
    def run_sequence(emit: Callable[[installer.Event], None]) -> None:
        token = read_token(context.home)
        windows: CatalogPort | None = None
        guest: CatalogPort | None = None
        if token is not None:
            windows = HttpCatalog(
                engine_url(), token, where="the Windows engine"
            )
            guest = GuestCatalog(
                context.runner,
                CRUCIBLE_DISTRO,
                token,
                ENGINE_PORT,
                where=f'the "{CRUCIBLE_DISTRO}" engine',
            )
        installer.EngineInstall(
            context.runner,
            emit,
            release=context.release,
            home=context.home,
            install_sh_url=INSTALL_SH_URL.format(release=context.release),
            windows_catalog=windows,
            guest_catalog=guest,
        ).run()

    return run_sequence


def _write_pairing(context: HostContext) -> None:
    """3.6: the Windows-side pairing file, ACL'd to this user.

    Written from the CONFIG the running server has, not composed: the name,
    the port and the token are all its, and a second composer of a pairing
    line is a second owner of the format.
    """
    from ..pairing import PairingFileError, pairing_line, write_pairing_file

    import tomllib

    path = context.home / "config.toml"
    if not path.is_file():
        context.log.write("pairing: no config yet, so no pairing file")
        return
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
        server = document["server"]
        line = pairing_line(
            server["name"],
            engine_url(),
            document["auth"]["token"],
        )
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        context.log.write(f"pairing: {path} could not be read ({exc})")
        return
    try:
        written = write_pairing_file(context.home, line, env=context.runner.env)
    except PairingFileError as exc:
        context.log.write(f"pairing: {exc.code}: {exc.message}")
        return
    context.log.write(f"pairing: {written}")
