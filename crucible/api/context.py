from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, FastAPI

from ..backend import Backend
from ..config import Config
from ..errors import ApiError
from ..leases import Leases
from ..residency import Residency


@dataclass(frozen=True)
class Routers:
    public: APIRouter
    private: APIRouter
    openai: APIRouter
    peer: APIRouter


@dataclass(frozen=True)
class AppContext:
    app: FastAPI
    config: Config
    backend: Backend
    residency: Residency
    leases: Leases
    registry: dict[str, Any]
    decide_here: Callable[[str], ApiError]
