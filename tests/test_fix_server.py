"""Server hardening: path-import roots, CORS, upload limits, HEAD routes, person patching,
library visibility by media status, identity constraints against stored sightings, and the
job manager's bookkeeping (progress, event memory, shutdown, spool cleanup)."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import httpx
import pytest

import ovp.jobs as jobs_mod
from ovp.api.app import app
from ovp.config import get_settings
from ovp.jobs import InputItem, JobManager, PathNotAllowed, iter_media_files
from ovp.pipeline.analyzer import AnalyzeOptions, Cancelled
from ovp.services.identity import IdentityService, presence_segments, segments_overlap
from ovp.storage.db import Database

from .fakes import BLUE, RED, make_image, png_bytes
from .seed import SPACE, add_entities, add_media, entity_rows, person_ids, vec

A = vec(1, 0, 0, 0)
B = vec(0, 1, 0, 0)


@pytest.fixture
async def client(app_ctx: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    s = get_settings()
    for name in ("import_roots", "max_upload_mb"):
        monkeypatch.setattr(s, name, getattr(s, name))  # restored after the test
    return s


def _scene(*colors: tuple[int, int, int]) -> bytes:
    return png_bytes(make_image([(c, (20 + 90 * i, 30, 90 + 90 * i, 200)) for i, c in enumerate(colors)]))


# ── server-side path import ───────────────────────────────────────────────────


def _tree(root: Path) -> None:
    (root / "sub").mkdir(parents=True)
    for name in ("a.jpg", "sub/b.PNG", "sub/c.mp4", "notes.txt"):
        (root / name).write_bytes(b"x")


async def test_path_import_disabled_without_roots(client: httpx.AsyncClient, settings: Any, tmp_path: Path) -> None:
    settings.import_roots = []
    _tree(tmp_path / "lib")
    r = await client.post("/api/analyze/paths", json={"paths": [str(tmp_path / "lib")]})
    assert r.status_code == 403 and "OVP_IMPORT_ROOTS" in r.json()["detail"]


async def test_path_import_confined_to_roots(app_ctx: Any, client: httpx.AsyncClient, settings: Any, tmp_path: Path) -> None:
    lib, outside = tmp_path / "lib", tmp_path / "private"
    _tree(lib)
    _tree(outside)
    (lib / "escape").symlink_to(outside, target_is_directory=True)
    (lib / "escape.jpg").symlink_to(outside / "a.jpg")
    settings.import_roots = [lib.resolve()]

    for bad in (outside, lib / ".." / "private", lib / "escape", lib / "escape.jpg", Path("/")):
        r = await client.post("/api/analyze/paths", json={"paths": [str(bad)]})
        assert r.status_code == 403, bad

    items = await app_ctx.jobs.collect_paths([str(lib)])
    # symlinks met while walking are skipped (the folder link and the file link)
    assert [(i.name, i.kind, i.temporary) for i in items] == [("a.jpg", "image", False), ("b.PNG", "image", False), ("c.mp4", "video", False)]
    assert len(await app_ctx.jobs.collect_paths([str(lib)], max_items=2)) == 2
    # the local CLI is not restricted
    settings.import_roots = []
    assert len(await app_ctx.jobs.collect_paths([str(outside)], restricted=False)) == 3


def test_iter_media_files_is_lazy(tmp_path: Path) -> None:
    _tree(tmp_path)
    it = iter_media_files([str(tmp_path)], max_items=10)
    assert next(it).name == "a.jpg"  # yields before walking the rest
    with pytest.raises(PathNotAllowed):
        list(iter_media_files([str(tmp_path)], 10, roots=[tmp_path / "sub"]))


# ── HTTP hardening ────────────────────────────────────────────────────────────


async def test_cors_only_for_configured_origins(client: httpx.AsyncClient) -> None:
    pre = {"Access-Control-Request-Method": "POST"}
    evil = await client.options("/api/analyze/paths", headers={"Origin": "https://evil.example", **pre})
    assert "access-control-allow-origin" not in evil.headers
    dev = await client.options("/api/analyze/paths", headers={"Origin": "http://localhost:5173", **pre})
    assert dev.headers["access-control-allow-origin"] == "http://localhost:5173"


async def test_upload_size_limit(app_ctx: Any, client: httpx.AsyncClient, settings: Any) -> None:
    settings.max_upload_mb = 1
    big = b"\0" * (2 << 20)
    r = await client.post("/api/analyze", files=[("files", ("big.png", big, "image/png"))])
    assert r.status_code == 413 and "OVP_MAX_UPLOAD_MB" in r.json()["detail"]

    # without a Content-Length the handler enforces the limit while it reads
    boundary = "xyzzy"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"big.mp4\"\r\nContent-Type: video/mp4\r\n\r\n".encode()
        + big + f"\r\n--{boundary}--\r\n".encode()
    )

    async def chunks() -> AsyncIterator[bytes]:
        for i in range(0, len(body), 1 << 16):
            yield body[i:i + (1 << 16)]

    r = await client.post("/api/analyze", content=chunks(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert r.status_code == 413 and list(app_ctx.jobs.tmp_dir.iterdir()) == []  # partial spool removed


async def test_head_on_file_routes(client: httpx.AsyncClient) -> None:
    files = [("files", ("s.png", _scene(RED, BLUE), "image/png"))]
    out = (await client.post("/api/analyze?wait=true", files=files, data={"options": json.dumps({"query": "people"})})).json()
    (media,) = out["media"]
    urls = [media["url"], media["thumb_url"], media["entities"][0]["crop_url"]]
    for url in urls:
        r = await client.head(url)
        assert r.status_code == 200 and int(r.headers["content-length"]) > 0 and r.content == b"", url


# ── persons ───────────────────────────────────────────────────────────────────


async def test_patch_person_semantics(client: httpx.AsyncClient, db: Database) -> None:
    pid = await db.create_person(SPACE, A, None, label="Alice")
    await db.update_person(pid, notes="gate")
    r = await client.patch(f"/api/persons/{pid}", json={"is_watchlist": True})
    assert r.status_code == 200 and r.json()["label"] == "Alice" and r.json()["is_watchlist"] is True
    assert (await client.patch(f"/api/persons/{pid}", json={"is_watchlist": None})).status_code == 422
    p = (await client.patch(f"/api/persons/{pid}", json={"label": "", "notes": None})).json()
    assert p["label"] is None and p["notes"] is None and p["is_watchlist"] is True
    assert (await client.patch("/api/persons/p_missing", json={"label": "x"})).status_code == 404


async def test_person_search_escapes_like_and_exact_label(db: Database) -> None:
    bob = await db.create_person(SPACE, A, None, label="Bob")
    await db.create_person(SPACE, B, None, label="Bobby")
    await db.create_person(SPACE, B, None, label="100% sure")
    assert (await db.list_persons(q="B_b"))[1] == 0  # '_' is not a wildcard
    assert [r["label"] for r in (await db.list_persons(q="%"))[0]] == ["100% sure"]
    assert [r["id"] for r in await db.find_persons_by_label(" bob ")] == [bob]


async def test_moving_only_sighting_away_deletes_empty_person(client: httpx.AsyncClient, db: Database) -> None:
    identity = IdentityService(db)
    mid = await add_media(db)
    (eid,) = await add_entities(db, mid, [A])
    res = await identity.assign_media([(eid, A)], SPACE)
    old = res[eid]["person_id"]
    r = await client.post(f"/api/entities/{eid}/assign", json={"person_id": None})
    assert r.status_code == 200
    new = r.json()["person"]["id"]
    assert new != old and await person_ids(db) == {new}
    assert (await client.get("/api/persons")).json()["total"] == 1


async def test_merge_and_delete_go_through_identity_lock(app_ctx: Any, client: httpx.AsyncClient, db: Database) -> None:
    keep = await db.create_person(SPACE, A, None, label="K")
    other = await db.create_person(SPACE, A, None, label="O")
    await app_ctx.identity.lock.acquire()
    merge = asyncio.create_task(client.post(f"/api/persons/{keep}/merge", json={"other_id": other}))
    await asyncio.sleep(0.05)
    assert not merge.done()  # waits for a running assignment/recluster
    app_ctx.identity.lock.release()
    assert (await merge).status_code == 200 and await person_ids(db) == {keep}
    assert (await client.delete(f"/api/persons/{keep}")).status_code == 200 and await person_ids(db) == set()


# ── identity against stored sightings ─────────────────────────────────────────


async def test_scan_cannot_link_two_people_of_one_image_to_one_person(db: Database) -> None:
    identity = IdentityService(db)
    mid = await add_media(db)
    (first,) = await add_entities(db, mid, [A])
    pa = (await identity.assign_media([(first, A)], SPACE))[first]["person_id"]
    # a later scan adds a second, look-alike person to the same image
    (second,) = await add_entities(db, mid, [A])
    res = await identity.assign_media([(second, A)], SPACE)
    assert res[second]["person_id"] not in (None, pa) and res[second]["status"] == "new"


async def test_video_stored_tracks_conflict_only_when_visible_together(db: Database) -> None:
    identity = IdentityService(db)
    mid = await add_media(db, kind="video")
    (first,) = await add_entities(db, mid, [A], spans=[(0.0, 4.0)], extra={"trajectory": [[t, 0, 0, 1, 1] for t in (0.0, 1.0, 2.0, 3.0, 4.0)]})
    pa = (await identity.assign_media([(first, A)], SPACE))[first]["person_id"]
    (later, overlapping) = await add_entities(db, mid, [A, A], spans=[(10.0, 12.0), (3.0, 6.0)])
    res = await identity.assign_media([(later, A), (overlapping, A)], SPACE, conflicts=lambda a, b: False)
    assert res[later]["person_id"] == pa
    assert res[overlapping]["person_id"] != pa
    rows = await entity_rows(db, [first, later])
    assert rows[first]["person_id"] == rows[later]["person_id"] == pa


def test_presence_segments() -> None:
    traj = [[t] for t in (0.0, 0.5, 1.0, 5.0, 5.5)]
    assert presence_segments(traj, 0.0, 5.5) == [(0.0, 1.0), (5.0, 5.5)]
    assert presence_segments(None, 2.0, None) == [(2.0, 2.0)]
    assert not segments_overlap([(0.0, 1.0), (5.0, 5.5)], [(2.0, 4.0)])
    assert segments_overlap([(0.0, 1.0)], [(1.0, 2.0)])


# ── library visibility ────────────────────────────────────────────────────────


async def test_hidden_statuses_and_interrupted_media(client: httpx.AsyncClient, db: Database) -> None:
    ids = {}
    for status in ("done", "processing", "pending", "error", "cancelled"):
        ids[status] = await add_media(db)
        await db.update_media(ids[status], status=status)
    listing = (await client.get("/api/media")).json()
    assert {m["id"] for m in listing["media"]} == {ids["done"], ids["processing"]}
    assert listing["total"] == 1
    assert (await client.get("/api/stats")).json()["media"] == 1
    assert await db.all_media_ids() == [ids["done"]]
    assert await db.fail_interrupted_media() == 1
    assert (await db.get_media(ids["processing"]))["status"] == "error"


# ── job manager ───────────────────────────────────────────────────────────────


class StubAnalyzer:
    """Analyzer stand-in whose per-item behaviour the test controls."""

    def __init__(self, store: Any, db: Database):
        self.store, self.db = store, db
        self.gate = asyncio.Event()
        self.scan_results: dict[str, dict[str, int]] = {}
        self.video_calls: list[dict[str, Any]] = []

    async def make_plan(self, opts: AnalyzeOptions) -> Any:
        return SimpleNamespace(to_dict=lambda: {"prompts": []})

    async def analyze_image(self, img: Any, name: str, opts: Any, plan: Any, emit: Any, cancel: threading.Event) -> str:
        await self.gate.wait()
        if cancel.is_set():
            raise Cancelled()
        return await add_media(self.db, name=name)

    async def analyze_video(self, path: Path, name: str, opts: Any, plan: Any, emit: Any, cancel: Any = None, move: bool = False) -> str:
        self.video_calls.append({"name": name, "move": move})
        return await add_media(self.db, kind="video", name=name)

    async def scan_media(self, media_id: str, opts: Any, plan: Any, emit: Any, cancel: Any = None) -> dict[str, int]:
        return self.scan_results[media_id]


@pytest.fixture
def manager(db: Database, store: Any) -> JobManager:
    return JobManager(StubAnalyzer(store, db), db, store)  # type: ignore[arg-type]


def _spool(manager: JobManager, name: str, data: bytes = b"x") -> InputItem:
    p = manager.tmp_path(name)
    p.write_bytes(data)
    return InputItem(p, name, "video" if name.endswith(".mp4") else "image")


async def test_cancelled_item_not_counted_as_done(manager: JobManager) -> None:
    img = _spool(manager, "a.png", _scene(RED))
    job = manager.submit_analyze([img], AnalyzeOptions())
    await asyncio.sleep(0.05)
    manager.cancel(job)
    manager.analyzer.gate.set()  # type: ignore[attr-defined]
    await manager.wait(job)
    assert job.status == "cancelled" and job.progress["done"] == 0 and job.progress["total"] == 1
    assert not img.path.exists()


async def test_videos_are_moved_and_spool_is_cleaned(manager: JobManager) -> None:
    vid = _spool(manager, "v.mp4")
    job = manager.submit_analyze([vid], AnalyzeOptions())
    await manager.wait(job)
    assert job.status == "done" and manager.analyzer.video_calls == [{"name": "v.mp4", "move": True}]  # type: ignore[attr-defined]
    assert not vid.path.exists()


async def test_queued_job_cancelled_at_shutdown_removes_spooled_uploads(db: Database, store: Any) -> None:
    manager = JobManager(StubAnalyzer(store, db), db, store)  # type: ignore[arg-type]
    # occupy every job slot (OVP_JOB_CONCURRENCY) with blocked jobs
    blockers = [manager.submit_analyze([_spool(manager, f"b{i}.png", _scene(RED))], AnalyzeOptions()) for i in range(get_settings().job_concurrency)]
    queued_item = _spool(manager, "q.png", _scene(RED))
    queued = manager.submit_analyze([queued_item], AnalyzeOptions())
    await asyncio.sleep(0.05)
    assert queued.status == "queued"
    shutdown = asyncio.create_task(manager.shutdown(timeout=5))
    await asyncio.sleep(0.05)
    manager.analyzer.gate.set()  # type: ignore[attr-defined]
    await shutdown
    assert all(j.status == "cancelled" for j in (*blockers, queued))
    assert not queued_item.path.exists() and list(manager.tmp_dir.iterdir()) == []


async def test_shutdown_cancels_stuck_tasks(manager: JobManager) -> None:
    job = manager.submit_analyze([_spool(manager, "a.png", _scene(RED))], AnalyzeOptions())
    await asyncio.sleep(0.05)
    await manager.shutdown(timeout=0.1)  # the stub ignores the cancel flag until the gate opens
    assert job.task is not None and job.task.done() and job.status == "cancelled"


async def test_scan_summary_counts_only_accepted_hits(manager: JobManager, db: Database) -> None:
    hit, miss, updated = [await add_media(db) for _ in range(3)]
    manager.analyzer.scan_results = {  # type: ignore[attr-defined]
        hit: {"accepted": 2, "rejected": 1, "updated": 0},
        miss: {"accepted": 0, "rejected": 3, "updated": 0},
        updated: {"accepted": 0, "rejected": 0, "updated": 2},
    }
    job = manager.submit_scan(None, AnalyzeOptions(query="x"))
    await manager.wait(job)
    assert job.summary == {"scanned": 3, "media_with_hits": 1, "accepted": 2, "rejected": 4, "updated": 2}
    assert job.media_ids == [hit] and [e["media"]["id"] for e in job.events if e["type"] == "media_done"] == [hit]


async def test_media_details_keeps_order(manager: JobManager, db: Database) -> None:
    m1, m2 = await add_media(db), await add_media(db)
    await add_entities(db, m2, [A, B])
    out = await manager.media_details([m2, "m_missing", m1])
    assert [m["id"] for m in out] == [m2, m1] and len(out[0]["entities"]) == 2 and out[1]["entities"] == []


async def test_event_memory_is_bounded(manager: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs_mod, "MAX_EVENTS", 5)
    monkeypatch.setattr(jobs_mod, "KEEP_HISTORY", 1)
    monkeypatch.setattr(jobs_mod, "MAX_JOBS", 2)
    manager.analyzer.gate.set()  # type: ignore[attr-defined]
    done = []
    for i in range(3):
        job = manager.submit_analyze([_spool(manager, f"{i}.png", _scene(RED)) for _ in range(4)], AnalyzeOptions())
        await manager.wait(job)
        done.append(job)
        assert len(job.events) <= 6 and job.events[-1]["type"] == "job_done"
    assert done[0].id not in manager.jobs and len(manager.jobs) == 2
    assert [e["type"] for e in done[1].events] == ["job_done"] and len(done[2].events) > 1
    replay = [ev async for ev in manager.subscribe(done[1])]
    assert [e["type"] for e in replay] == ["job_done"]


async def test_slow_subscriber_is_dropped(manager: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs_mod, "SUBSCRIBER_QUEUE", 2)
    job = manager.submit_analyze([_spool(manager, f"{i}.png", _scene(RED)) for i in range(5)], AnalyzeOptions())
    stream = manager.subscribe(job)
    first = await stream.__anext__()  # subscribed; now stop reading while the job runs
    manager.analyzer.gate.set()  # type: ignore[attr-defined]
    await manager.wait(job)
    rest = [ev async for ev in stream]
    assert first and len(rest) <= 2 and not job.subscribers
