"""Kernel-held process locks; PID files are informational, never the mutex."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


class ProcessLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> bool:
        if self.handle is not None:
            return False
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
            kernel.CreateMutexW.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            name = "Local\\Crucible-" + hashlib.sha256(str(self.path.resolve()).casefold().encode()).hexdigest()
            handle = kernel.CreateMutexW(None, False, name)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if ctypes.get_last_error() == 183:  # object already held by another process
                kernel.CloseHandle(handle)
                return False
            self.handle = (kernel, handle)
        else:
            import fcntl
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                return False
            self.handle = handle
        return True

    def close(self) -> None:
        handle, self.handle = self.handle, None
        if handle is None:
            return
        if os.name == "nt":
            kernel, value = handle
            kernel.CloseHandle(value)
        else:
            handle.close()
