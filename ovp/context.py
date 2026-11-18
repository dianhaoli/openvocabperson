"""Application context: wires storage, models and services together (one per process)."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .agent.agent import Agent
from .config import get_settings
from .jobs import JobManager
from .models.registry import ModelRegistry, get_registry
from .pipeline.analyzer import Analyzer
from .pipeline.planner import QueryPlanner
from .services.identity import IdentityService
from .services.search import SearchService
from .storage.db import Database, get_db
from .storage.files import FileStore, get_store

log = logging.getLogger("ovp")


class AppContext:
    def __init__(self, db: Database, store: FileStore, registry: ModelRegistry):
        self.settings = get_settings()
        self.db, self.store, self.registry = db, store, registry
        self.planner = QueryPlanner()
        self.identity = IdentityService(db)
        self.search = SearchService(db, registry, self.planner)
        self.analyzer = Analyzer(db, store, registry, self.planner, self.identity)
        self.jobs = JobManager(self.analyzer, db, store)
        self.agent = Agent(self)
        self.ready = False
        self.warmup_task: Optional[asyncio.Task] = None

    def reid_space(self) -> Optional[str]:
        reid = self.registry.get("reid")
        return reid.space if reid is not None else None

    async def warmup(self) -> None:
        try:
            await self.registry.warmup_defaults()
        except Exception as e:  # never fatal: the API still serves the library
            log.warning("model warmup failed: %s", e)
        finally:
            self.ready = True
            log.info("models ready: %s", self.registry.defaults)


_ctx: Optional[AppContext] = None


async def create_context() -> AppContext:
    global _ctx
    _ctx = AppContext(await get_db(), get_store(), get_registry())
    return _ctx


def get_context() -> AppContext:
    if _ctx is None:
        raise RuntimeError("application context not initialised")
    return _ctx
