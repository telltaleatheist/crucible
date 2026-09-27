from __future__ import annotations

from enum import Enum, StrEnum


class Distro(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class Engine(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    INSTALLING = "installing"


class Owner(str, Enum):
    NONE = "none"
    WSL_UNIT = "wsl-unit"
    HOST_CHILD = "host-child"
    FOUND = "found"


class MoveState(StrEnum):
    DONE = "done"
    REBOOT_PENDING = "reboot-pending"
    CANNOT = "cannot"
    FAILED = "failed"
    DECLINED = "declined"


class EngineDecision(StrEnum):
    DONE = "done"
    REBOOT_PENDING = "reboot-pending"
    CANNOT = "cannot"
    FAILED = "failed"
    DECLINED = "declined"
    FOUND = "found"
    UNREADABLE = "unreadable"
    NO_DOOR = "no_door"
    ALREADY_RUNNING = "already_running"

    @classmethod
    def of(cls, state: MoveState) -> "EngineDecision":
        return cls(state.value)
