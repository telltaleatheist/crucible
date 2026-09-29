from __future__ import annotations

import secrets
import socket
import threading
import time
from pathlib import Path
from typing import Callable

from ..processlock import ProcessLock

LOCK_NAME = "app.lock"
DOOR_NAME = "app.door"

FOCUS = "focus"
QUIT = "quit"
WORDS = (FOCUS, QUIT)

CONNECT_SECONDS = 3.0
CLOSE_SECONDS = 15.0
POLL_SECONDS = 0.1


def door_path(home: Path) -> Path:
    return Path(home) / DOOR_NAME


def read_door(home: Path) -> tuple[int, str] | None:
    try:
        port_text, token = door_path(home).read_text(encoding="ascii").split()
        return int(port_text), token
    except (OSError, ValueError):
        return None


def signal(home: Path, word: str) -> bool:
    door = read_door(home)
    if door is None:
        return False
    port, token = door
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=CONNECT_SECONDS) as link:
            link.sendall(f"{token} {word}\n".encode("ascii"))
            return link.recv(8) == b"ok\n"
    except OSError:
        return False


class Instance:
    def __init__(self, home: Path) -> None:
        self.home = Path(home)
        self.lock = ProcessLock(self.home / LOCK_NAME)
        self.token = secrets.token_hex(16)
        self.server: socket.socket | None = None

    def claim(self) -> bool:
        self.home.mkdir(parents=True, exist_ok=True)
        return self.lock.acquire()

    def serve(self, heard: Callable[[str], None]) -> int:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(4)
        self.server = server
        port = server.getsockname()[1]
        door_path(self.home).write_text(f"{port} {self.token}\n", encoding="ascii")
        threading.Thread(target=self._accept, args=(server, heard), daemon=True).start()
        return port

    def _accept(self, server: socket.socket, heard: Callable[[str], None]) -> None:
        while True:
            try:
                link, _ = server.accept()
            except OSError:
                return
            with link:
                self._answer(link, heard)

    def _answer(self, link: socket.socket, heard: Callable[[str], None]) -> None:
        link.settimeout(CONNECT_SECONDS)
        try:
            token, _, word = link.recv(128).decode("ascii", "replace").strip().partition(" ")
        except OSError:
            return
        if not secrets.compare_digest(token, self.token) or word not in WORDS:
            return
        link.sendall(b"ok\n")
        heard(word)

    def close(self) -> None:
        if self.server is not None:
            self.server.close()
            self.server = None
            door = read_door(self.home)
            if door is not None and door[1] == self.token:
                door_path(self.home).unlink(missing_ok=True)
        self.lock.close()


def still_open(home: Path) -> bool:
    probe = ProcessLock(Path(home) / LOCK_NAME)
    if probe.acquire():
        probe.close()
        return False
    return True


def close_running(home: Path, timeout: float = CLOSE_SECONDS) -> bool:
    if not still_open(home):
        return True
    signal(home, QUIT)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not still_open(home):
            return True
        time.sleep(POLL_SECONDS)
    return False
