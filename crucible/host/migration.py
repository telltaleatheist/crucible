from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, ContextManager

from ..platform.errors import HostError
from . import cleanup_record, installer
from .catalog import CatalogPort, StoppedWindowsCatalog
from .context import HostContext
from .presence import Presence
from .state import Engine, Owner

NOT_READY = "migration_cleanup_not_ready"

RETRY_SECONDS = 300.0

GUEST_BACKEND = installer.GUEST_BACKEND


def verify_active_guest(
    presence: Presence, token: str | None, read_info: Callable[[str], dict[str, Any]]
) -> None:
    if presence.owner is not Owner.WSL_UNIT or presence.engine is not Engine.RUNNING:
        raise HostError(NOT_READY, "The WSL engine has not taken ownership; Windows models are kept")
    if token is None:
        raise HostError(NOT_READY, "The WSL engine has no pairing; Windows models are kept")
    info = read_info(token)
    if info.get("host", {}).get("backend") != GUEST_BACKEND:
        raise HostError(NOT_READY, "The authenticated endpoint is not the WSL engine; Windows models are kept")


def stopped_windows_catalog(home: Path) -> CatalogPort:
    from ..backend import detect_backend
    from ..config import load_config

    return StoppedWindowsCatalog(load_config(home), detect_backend(), cleanup_record.cleanup_subjects(home))


class ModelCleanup:
    def __init__(
        self,
        context: HostContext,
        *,
        lock: ContextManager[Any],
        windows_catalog: Callable[[], CatalogPort],
        guest_catalog: Callable[[], CatalogPort],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._context = context
        self._lock = lock
        self._windows_catalog = windows_catalog
        self._guest_catalog = guest_catalog
        self._clock = clock
        self.running = False
        self.retry_at = 0.0

    @property
    def record(self) -> Path:
        return self._context.home / cleanup_record.CLEANUP_RECORD

    def due(self) -> bool:
        presence = self._context.presence
        return (
            presence.owner is Owner.WSL_UNIT
            and presence.engine is Engine.RUNNING
            and self.record.exists()
            and not self.running
            and self._clock() >= self.retry_at
        )

    def start_in_background(self) -> None:
        self.running = True
        threading.Thread(target=self.resume, name="crucible-model-cleanup", daemon=True).start()

    def resume(self, *, raise_errors: bool = False) -> None:
        try:
            with self._lock:
                if self.record.exists() and not self._quarantined():
                    self._retire_windows_copies()
        except Exception as exc:
            self._context.log.write(f"model cleanup pending: {exc}")
            if raise_errors:
                raise
        finally:
            self.retry_at = self._clock() + RETRY_SECONDS
            self.running = False

    def _quarantined(self) -> bool:
        context = self._context
        return cleanup_record.quarantine_bad_cleanup_record(context.home, context.log.write) is not None

    def _retire_windows_copies(self) -> None:
        context = self._context
        windows = self._windows_catalog()
        guest = self._guest_catalog()
        walk = context.install_walk(
            context.logged_as("model cleanup"), windows_catalog=windows, guest_catalog=guest
        )
        walk.migrate_weights(allow_pull=False)
        self.record.unlink()
        context.log.write("model cleanup: completed; native runtime kept, migrated Windows model files removed")
