"""Background jobs: batch image/video analysis and library scans with live event streams.

* uploads are spooled to disk by the request handler; the job runs independently of the HTTP
  request, so closing the browser tab does not kill an analysis
* several media items of one job are processed concurrently (local models are still serialized
  on the device thread; cloud VLM calls overlap)
* every event is kept (bounded) so late subscribers get a full replay over SSE
* cancellation is cooperative (checked between stages and inside the video tracking loop)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from . import serialize
from .config import get_settings
from .pipeline.analyzer import AnalyzeOptions, Analyzer, Cancelled
from .storage.db import Database, new_id
from .storage.files import FileStore, media_kind

log = logging.getLogger("ovp.jobs")

MAX_EVENTS = 20000
MAX_JOBS = 200


@dataclass
class InputItem:
    path: Path
    name: str
    kind: str  # image | video
    temporary: bool = True


@dataclass
class Job:
    id: str
    kind: str
    options: dict[str, Any]
    status: str = "queued"
    progress: dict[str, Any] = field(default_factory=lambda: {"done": 0, "total": 0, "stage": None, "message": None})
    media_ids: list[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    cancel: threading.Event = field(default_factory=threading.Event)
    task: Optional[asyncio.Task] = None
    plan: Optional[dict[str, Any]] = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": dict(self.progress),
            "media_ids": list(self.media_ids),
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "options": self.options,
            "plan": self.plan,
            "summary": self.summary,
        }

    @property
    def finished(self) -> bool:
        return self.status in ("done", "error", "cancelled")


class JobManager:
    def __init__(self, analyzer: Analyzer, db: Database, store: FileStore):
        self.analyzer, self.db, self.store = analyzer, db, store
        self.jobs: dict[str, Job] = {}
        s = get_settings()
        self._sem = asyncio.Semaphore(s.job_concurrency)
        self.media_concurrency = s.media_concurrency
        self.tmp_dir = store.root / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── events ────────────────────────────────────────────────────────────────
    def emitter(self, job: Job):
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()

        def _emit(type: str, **data: Any) -> None:
            ev = {"type": type, "job_id": job.id, "ts": round(time.time(), 3), **data}
            if type == "stage":
                job.progress["stage"] = data.get("stage")
                job.progress["message"] = data.get("message")
            if len(job.events) < MAX_EVENTS or type in ("job_done", "media_done", "progress"):
                job.events.append(ev)
            for q in list(job.subscribers):
                q.put_nowait(ev)

        def emit(type: str, **data: Any) -> None:
            if threading.get_ident() == loop_thread:
                _emit(type, **data)
            else:
                loop.call_soon_threadsafe(lambda: _emit(type, **data))

        return emit

    async def subscribe(self, job: Job) -> AsyncIterator[dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue()
        backlog = list(job.events)
        job.subscribers.add(q)
        try:
            for ev in backlog:
                yield ev
            if job.finished and backlog and backlog[-1]["type"] == "job_done":
                return
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"type": "ping", "job_id": job.id, "ts": time.time()}
                    continue
                yield ev
                if ev["type"] == "job_done":
                    return
        finally:
            job.subscribers.discard(q)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def _new(self, kind: str, options: dict[str, Any]) -> Job:
        job = Job(id=new_id("j"), kind=kind, options=options)
        self.jobs[job.id] = job
        if len(self.jobs) > MAX_JOBS:
            for jid in sorted(self.jobs, key=lambda k: self.jobs[k].created_at)[: len(self.jobs) - MAX_JOBS]:
                if self.jobs[jid].finished:
                    self.jobs.pop(jid, None)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job: Job) -> None:
        if not job.finished:
            job.cancel.set()

    async def wait(self, job: Job, timeout: Optional[float] = None) -> Job:
        if job.task is not None:
            await asyncio.wait_for(asyncio.shield(job.task), timeout)
        return job

    def submit_analyze(self, items: list[InputItem], opts: AnalyzeOptions) -> Job:
        job = self._new("analyze", opts.to_dict())
        job.progress["total"] = len(items)
        job.task = asyncio.create_task(self._run(job, self._analyze(job, items, opts)))
        return job

    def submit_scan(self, media_ids: Optional[list[str]], opts: AnalyzeOptions, limit: Optional[int] = None) -> Job:
        job = self._new("scan", {**opts.to_dict(), "media_ids": media_ids, "limit": limit})
        job.task = asyncio.create_task(self._run(job, self._scan(job, media_ids, opts, limit)))
        return job

    async def _run(self, job: Job, coro: Any) -> None:
        emit = self.emitter(job)
        async with self._sem:
            job.status = "running"
            emit("job_started", job=job.to_dict())
            try:
                await coro
                job.status = "cancelled" if job.cancel.is_set() else "done"
            except Cancelled:
                job.status = "cancelled"
            except asyncio.CancelledError:
                job.status = "cancelled"
            except Exception as e:
                log.error("job %s failed: %s", job.id, traceback.format_exc())
                job.status, job.error = "error", str(e)
                emit("error", message=str(e))
            finally:
                job.finished_at = time.time()
                job.progress["stage"] = None
                emit("job_done", job=job.to_dict())

    # ── analyze ───────────────────────────────────────────────────────────────
    async def _analyze(self, job: Job, items: list[InputItem], opts: AnalyzeOptions) -> None:
        emit = self.emitter(job)
        try:
            plan = await self.analyzer.make_plan(opts)
            job.plan = plan.to_dict()
            emit("plan", plan=job.plan)
            sem = asyncio.Semaphore(self.media_concurrency)
            counts = {"entities": 0, "failed": 0}

            async def one(i: int, it: InputItem) -> None:
                async with sem:
                    if job.cancel.is_set():
                        return
                    emit("media_started", index=i, total=len(items), name=it.name, kind=it.kind)
                    try:
                        if it.kind == "video":
                            mid = await self.analyzer.analyze_video(it.path, it.name, opts, plan, emit, job.cancel)
                        else:
                            img = await self.store.open_image(it.path)
                            mid = await self.analyzer.analyze_image(img, it.name, opts, plan, emit, job.cancel)
                        job.media_ids.append(mid)
                        detail = await self._detail(mid)
                        counts["entities"] += len(detail["entities"]) if detail else 0
                        emit("media_done", media=detail)
                    except Cancelled:
                        raise
                    except Exception as e:
                        counts["failed"] += 1
                        log.error("analysis of %s failed: %s", it.name, traceback.format_exc())
                        emit("log", level="error", message=f"{it.name}: {e}")
                    finally:
                        job.progress["done"] += 1
                        emit("progress", done=job.progress["done"], total=job.progress["total"])

            results = await asyncio.gather(*(one(i, it) for i, it in enumerate(items)), return_exceptions=True)
            for r in results:
                if isinstance(r, Cancelled):
                    raise r
            job.summary = {"media": len(job.media_ids), **counts}
            if counts["failed"] and not job.media_ids:
                raise RuntimeError(f"all {counts['failed']} item(s) failed")
        finally:
            for it in items:
                if it.temporary:
                    try:
                        it.path.unlink(missing_ok=True)
                    except OSError:
                        pass

    async def _scan(self, job: Job, media_ids: Optional[list[str]], opts: AnalyzeOptions, limit: Optional[int]) -> None:
        emit = self.emitter(job)
        plan = await self.analyzer.make_plan(opts)
        job.plan = plan.to_dict()
        emit("plan", plan=job.plan)
        ids = media_ids or await self.db.all_media_ids(limit=limit)
        if limit:
            ids = ids[:limit]
        job.progress["total"] = len(ids)
        sem = asyncio.Semaphore(self.media_concurrency)
        found = {"entities": 0}

        async def one(i: int, mid: str) -> None:
            async with sem:
                if job.cancel.is_set():
                    return
                emit("media_started", index=i, total=len(ids), name=mid, kind=None)
                try:
                    n = await self.analyzer.scan_media(mid, opts, plan, emit, job.cancel)
                    found["entities"] += n
                    if n:
                        job.media_ids.append(mid)
                        emit("media_done", media=await self._detail(mid))
                except Cancelled:
                    raise
                except Exception as e:
                    emit("log", level="error", message=f"{mid}: {e}")
                finally:
                    job.progress["done"] += 1
                    emit("progress", done=job.progress["done"], total=len(ids))

        results = await asyncio.gather(*(one(i, m) for i, m in enumerate(ids)), return_exceptions=True)
        for r in results:
            if isinstance(r, Cancelled):
                raise r
        job.summary = {"scanned": len(ids), "media_with_hits": len(job.media_ids), **found}

    async def _detail(self, media_id: str) -> Optional[dict[str, Any]]:
        m = await self.db.get_media(media_id)
        if m is None:
            return None
        return serialize.media_detail(m, await self.db.media_entities(media_id))

    # ── inputs ────────────────────────────────────────────────────────────────
    def tmp_path(self, name: str) -> Path:
        suffix = Path(name).suffix.lower()[:8]
        return self.tmp_dir / f"{new_id('u')}{suffix}"

    @staticmethod
    def collect_paths(paths: list[str], max_items: int = 5000) -> list[InputItem]:
        out: list[InputItem] = []
        for raw in paths:
            p = Path(raw).expanduser()
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    k = media_kind(f.name)
                    if f.is_file() and k:
                        out.append(InputItem(f, f.name, k, temporary=False))
            elif p.is_file() and media_kind(p.name):
                out.append(InputItem(p, p.name, media_kind(p.name) or "image", temporary=False))
            if len(out) >= max_items:
                break
        return out[:max_items]
