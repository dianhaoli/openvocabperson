"""Background jobs: batch image/video analysis and library scans with live event streams.

* uploads are spooled to disk by the request handler; the job runs independently of the HTTP
  request, so closing the browser tab does not kill an analysis
* several media items of one job are processed concurrently (local models are still serialized
  on the device thread; cloud VLM calls overlap)
* events are kept so late subscribers get a replay over SSE; the history is bounded per job
  and dropped (except the final ``job_done``) for finished jobs beyond the newest few
* jobs live in memory only: they are lost on restart (their media stay in the library)
* cancellation is cooperative (checked between stages and inside the video tracking loop)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import time
import traceback
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Optional, Sequence

from . import serialize
from .config import get_settings
from .pipeline.analyzer import AnalyzeOptions, Analyzer, Cancelled
from .storage.db import Database, new_id
from .storage.files import FileStore, media_kind

log = logging.getLogger("ovp.jobs")

MAX_EVENTS = 5000  # stored events per job; job_done is always kept
MAX_JOBS = 100  # finished jobs beyond this are forgotten
KEEP_HISTORY = 20  # finished jobs (newest first) that keep their full event history
SUBSCRIBER_QUEUE = 2000  # a subscriber this far behind is disconnected (it can reconnect and replay)


class PathNotAllowed(PermissionError):
    """A server-side import path is outside OVP_IMPORT_ROOTS (or path imports are disabled)."""


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
        self._media_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
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
            if len(job.events) < MAX_EVENTS or type == "job_done":
                job.events.append(ev)
            for q in list(job.subscribers):
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    job.subscribers.discard(q)  # its stream ends once drained

        def emit(type: str, **data: Any) -> None:
            if threading.get_ident() == loop_thread:
                _emit(type, **data)
            else:
                loop.call_soon_threadsafe(lambda: _emit(type, **data))

        return emit

    async def subscribe(self, job: Job) -> AsyncIterator[dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue(SUBSCRIBER_QUEUE)
        backlog = list(job.events)
        job.subscribers.add(q)
        try:
            for ev in backlog:
                yield ev
            if job.finished and backlog and backlog[-1]["type"] == "job_done":
                return
            while True:
                if q.empty() and q not in job.subscribers:
                    return  # dropped for falling behind
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
        self._trim()
        return job

    def _trim(self) -> None:
        """Forget the oldest finished jobs beyond MAX_JOBS and free the event history of finished
        jobs beyond the newest KEEP_HISTORY (their replay is just the final job_done)."""
        finished = [j for j in self.recent() if j.finished]
        for j in finished[KEEP_HISTORY:]:
            if len(j.events) > 1:
                j.events = j.events[-1:]
        excess = len(self.jobs) - MAX_JOBS
        for j in reversed(finished):
            if excess <= 0:
                break
            self.jobs.pop(j.id, None)
            excess -= 1

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def recent(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job: Job) -> None:
        if not job.finished:
            job.cancel.set()

    async def wait(self, job: Job, timeout: Optional[float] = None) -> Job:
        if job.task is not None:
            await asyncio.wait_for(asyncio.shield(job.task), timeout)
        return job

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Cancel every job and wait (up to ``timeout``) for their tasks, so nothing touches the
        database or the models after they are closed."""
        tasks = [j.task for j in self.jobs.values() if j.task is not None and not j.task.done()]
        for job in self.jobs.values():
            self.cancel(job)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.wait(pending, timeout=2.0)

    def purge_tmp(self) -> None:
        """Remove upload spool files left over by a previous run (call at startup)."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def submit_analyze(self, items: list[InputItem], opts: AnalyzeOptions) -> Job:
        job = self._new("analyze", opts.to_dict())
        job.progress["total"] = len(items)
        spooled = [it.path for it in items if it.temporary]
        job.task = asyncio.create_task(self._run(job, lambda: self._analyze(job, items, opts), spooled))
        return job

    def submit_scan(self, media_ids: Optional[list[str]], opts: AnalyzeOptions, limit: Optional[int] = None) -> Job:
        job = self._new("scan", {**opts.to_dict(), "media_ids": media_ids, "limit": limit})
        job.task = asyncio.create_task(self._run(job, lambda: self._scan(job, media_ids, opts, limit)))
        return job

    async def _run(self, job: Job, work: Callable[[], Awaitable[None]], spooled: Sequence[Path] = ()) -> None:
        emit = self.emitter(job)
        try:
            async with self._sem:
                job.status = "running"
                emit("job_started", job=job.to_dict())
                try:
                    if not job.cancel.is_set():
                        await work()
                    job.status = "cancelled" if job.cancel.is_set() else "done"
                except (Cancelled, asyncio.CancelledError):
                    job.status = "cancelled"
                except Exception as e:
                    log.error("job %s failed: %s", job.id, traceback.format_exc())
                    job.status, job.error = "error", str(e)
                    emit("error", message=str(e))
        except asyncio.CancelledError:  # cancelled while queued
            job.status = "cancelled"
        finally:
            # spooled uploads are removed whether the job ran, failed or never started
            # (an analysed video was moved into storage already)
            for p in spooled:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            job.finished_at = time.time()
            job.progress["stage"] = None
            emit("job_done", job=job.to_dict())
            self._trim()

    # ── analyze ───────────────────────────────────────────────────────────────
    async def _analyze(self, job: Job, items: list[InputItem], opts: AnalyzeOptions) -> None:
        emit = self.emitter(job)
        plan = await self.analyzer.make_plan(opts)
        job.plan = plan.to_dict()
        emit("plan", plan=job.plan)
        sem = asyncio.Semaphore(self.media_concurrency)
        counts = {"entities": 0, "failed": 0}

        def advance() -> None:  # cancelled items are not counted as done
            job.progress["done"] += 1
            emit("progress", done=job.progress["done"], total=job.progress["total"])

        async def one(i: int, it: InputItem) -> None:
            async with sem:
                if job.cancel.is_set():
                    return
                emit("media_started", index=i, total=len(items), name=it.name, kind=it.kind)
                try:
                    if it.kind == "video":
                        mid = await self.analyzer.analyze_video(it.path, it.name, opts, plan, emit, job.cancel, move=it.temporary)
                    else:
                        img = await self.store.open_image(it.path)
                        mid = await self.analyzer.analyze_image(img, it.name, opts, plan, emit, job.cancel)
                except Cancelled:
                    raise
                except Exception as e:
                    counts["failed"] += 1
                    log.error("analysis of %s failed: %s", it.name, traceback.format_exc())
                    emit("log", level="error", message=f"{it.name}: {e}")
                    advance()
                    return
                job.media_ids.append(mid)
                detail = await self._detail(mid)
                counts["entities"] += len(detail["entities"]) if detail else 0
                emit("media_done", media=detail)
                advance()

        results = await asyncio.gather(*(one(i, it) for i, it in enumerate(items)), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                raise r
        job.summary = {"media": len(job.media_ids), **counts}
        if counts["failed"] and not job.media_ids:
            raise RuntimeError(f"all {counts['failed']} item(s) failed")

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
        found = {"accepted": 0, "rejected": 0, "updated": 0}

        async def one(i: int, mid: str) -> None:
            async with sem:
                if job.cancel.is_set():
                    return
                emit("media_started", index=i, total=len(ids), name=mid, kind=None)
                try:
                    # two jobs scanning one media at once would both store the same new entities
                    async with self._media_lock(mid):
                        res = await self.analyzer.scan_media(mid, opts, plan, emit, job.cancel)
                    for k in found:
                        found[k] += res[k]
                    if res["accepted"]:
                        job.media_ids.append(mid)
                        emit("media_done", media=await self._detail(mid))
                except Cancelled:
                    raise
                except Exception as e:
                    log.error("scan of %s failed: %s", mid, traceback.format_exc())
                    emit("log", level="error", message=f"{mid}: {e}")
                job.progress["done"] += 1
                emit("progress", done=job.progress["done"], total=len(ids))

        results = await asyncio.gather(*(one(i, m) for i, m in enumerate(ids)), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                raise r
        job.summary = {"scanned": len(ids), "media_with_hits": len(job.media_ids), **found}

    def _media_lock(self, media_id: str) -> asyncio.Lock:
        lock = self._media_locks.get(media_id)
        if lock is None:
            lock = self._media_locks[media_id] = asyncio.Lock()
        return lock

    async def _detail(self, media_id: str) -> Optional[dict[str, Any]]:
        m = await self.db.get_media(media_id)
        if m is None:
            return None
        return serialize.media_detail(m, await self.db.media_entities(media_id))

    async def media_details(self, media_ids: Sequence[str]) -> list[dict[str, Any]]:
        """MediaDetail payloads for ``media_ids`` in that order (two queries; missing ids skipped)."""
        if not media_ids:
            return []
        media = {m["id"]: m for m in await self.db.get_media_many(media_ids)}
        entities: dict[str, list[Any]] = {}
        for e in await self.db.media_entities_many(list(media)):
            entities.setdefault(e["media_id"], []).append(e)
        return [serialize.media_detail(media[mid], entities.get(mid, [])) for mid in media_ids if mid in media]

    # ── inputs ────────────────────────────────────────────────────────────────
    def tmp_path(self, name: str) -> Path:
        suffix = Path(name).suffix.lower()[:8]
        return self.tmp_dir / f"{new_id('u')}{suffix}"

    async def collect_paths(self, paths: list[str], max_items: int = 5000, restricted: bool = True) -> list[InputItem]:
        """Images and videos at server-side ``paths`` (files or folders, recursively), at most
        ``max_items``. The walk runs in a worker thread. ``restricted`` (API and agent) only
        allows paths inside OVP_IMPORT_ROOTS and raises PathNotAllowed otherwise; the local CLI
        passes ``restricted=False``."""
        roots = get_settings().import_roots if restricted else None
        if roots is not None and not roots:
            raise PathNotAllowed("server-side path import is disabled: set OVP_IMPORT_ROOTS to the folders it may read")
        return await asyncio.to_thread(lambda: list(iter_media_files(paths, max_items, roots)))


def iter_media_files(paths: Sequence[str], max_items: int, roots: Optional[Sequence[Path]] = None) -> Iterator[InputItem]:
    """Lazily yield media files under ``paths`` (sorted per folder, stopping at ``max_items``).

    With ``roots``, every path is resolved first (following symlinks) and must lie inside one
    of them; symlinks found while walking are skipped, so a walk cannot leave its root.
    """
    n = 0
    resolved = []
    for raw in paths:
        p = Path(raw).expanduser()
        if roots is not None:
            p = p.resolve()
            if not any(p == r or p.is_relative_to(r) for r in roots):
                raise PathNotAllowed(f"{raw} is outside the allowed import folders")
        resolved.append(p)
    for p in resolved:
        if p.is_dir():
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames.sort()
                for name in sorted(filenames):
                    f = Path(dirpath) / name
                    k = media_kind(name)
                    if k and not (roots is not None and f.is_symlink()) and f.is_file():
                        yield InputItem(f, name, k, temporary=False)
                        n += 1
                        if n >= max_items:
                            return
        elif p.is_file() and (k := media_kind(p.name)):
            yield InputItem(p, p.name, k, temporary=False)
            n += 1
            if n >= max_items:
                return
