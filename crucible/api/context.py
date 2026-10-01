from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

import httpx
from fastapi import APIRouter, FastAPI

from .. import catalog, upstreams
from .. import peer as peer_module
from .. import settings as settings_module
from ..admission import AdmissionContext
from ..backend import Backend
from ..config import Config
from ..connect import PairingRequests
from ..errors import ApiError
from ..events import EventHub
from ..inflight import InFlight
from ..installonsubmit import InstallOnSubmit
from ..jobs.line import WaitingLine
from ..jobs.queue import JobStore
from ..leases import Leases
from ..residency import Residency
from ..settle import Settlement
from ..tasks import TaskStore
from ..ttsstream import StreamManager


@dataclass(frozen=True)
class Routers:
    public: APIRouter
    private: APIRouter
    openai: APIRouter
    peer: APIRouter


@dataclass(frozen=True)
class Services:
    events: EventHub
    leases: Leases
    store: JobStore
    line: WaitingLine
    streams: StreamManager
    inflight: InFlight
    ollama_contexts: upstreams.OllamaContexts
    settings_history: settings_module.History
    removals: catalog.Removals
    peer: peer_module.PeerState
    pairing_requests: PairingRequests
    settlement: Settlement
    tasks: TaskStore
    installs: InstallOnSubmit

    def publish(self, app: FastAPI) -> None:
        for one in fields(self):
            setattr(app.state, one.name, getattr(self, one.name))


@dataclass(frozen=True)
class AppContext:
    app: FastAPI
    config: Config
    backend: Backend
    residency: Residency
    decide_here: Callable[[str], ApiError]

    @property
    def events(self) -> EventHub:
        return self.app.state.events

    @property
    def store(self) -> JobStore:
        return self.app.state.store

    @property
    def line(self) -> WaitingLine:
        return self.app.state.line

    @property
    def tasks(self) -> TaskStore:
        return self.app.state.tasks

    @property
    def streams(self) -> StreamManager:
        return self.app.state.streams

    @property
    def leases(self) -> Leases:
        return self.app.state.leases

    @property
    def settlement(self) -> Settlement:
        return self.app.state.settlement

    @property
    def inflight(self) -> InFlight:
        return self.app.state.inflight

    @property
    def installs(self) -> InstallOnSubmit:
        return self.app.state.installs

    @property
    def peer(self) -> peer_module.PeerState:
        return self.app.state.peer

    @property
    def removals(self) -> catalog.Removals:
        return self.app.state.removals

    @property
    def settings_history(self) -> settings_module.History:
        return self.app.state.settings_history

    @property
    def pairing_requests(self) -> PairingRequests:
        return self.app.state.pairing_requests

    @property
    def ollama_contexts(self) -> upstreams.OllamaContexts:
        return self.app.state.ollama_contexts

    @property
    def http(self) -> httpx.AsyncClient:
        return self.app.state.http

    @property
    def started_at(self) -> float:
        return self.app.state.started_at

    @property
    def bind_host(self) -> str:
        return self.app.state.bind_host

    @property
    def bind_port(self) -> int:
        return self.app.state.bind_port

    def admission(self) -> AdmissionContext:
        return AdmissionContext(
            config=self.config,
            store=self.store,
            residency=self.residency,
            leases=self.leases,
            installs=self.installs,
            decide_here=self.decide_here,
            chats_in_flight=lambda: len(self.inflight),
            lease_granted=self.settlement.arm_for_lease_expiry,
        )
