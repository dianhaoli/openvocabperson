from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ovp.jobs import InputItem, Job, JobManager
from ovp.pipeline.analyzer import AnalyzeOptions

from .fakes import BLUE, GREEN, RED, FakeVLM, make_image, png_bytes


def _upload(tmp_path: Path, name: str, data: bytes) -> InputItem:
    path = tmp_path / f"upload_{name}"
    path.write_bytes(data)
    return InputItem(path, name, "image")


def _scene(*colors: tuple[int, int, int]) -> bytes:
    return png_bytes(make_image([(c, (20 + 90 * i, 30, 90 + 90 * i, 200)) for i, c in enumerate(colors)]))


async def _wait_for(job: Job, event_type: str) -> None:
    for _ in range(500):
        if any(e["type"] == event_type for e in job.events):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"no {event_type} event")


async def test_analyze_job_events_and_partial_failure(app_ctx: Any, tmp_path: Path) -> None:
    jobs: JobManager = app_ctx.jobs
    good = _upload(tmp_path, "scene.png", _scene(RED, GREEN, BLUE))
    broken = _upload(tmp_path, "broken.jpg", b"not an image")
    job = jobs.submit_analyze([good, broken], AnalyzeOptions())
    await jobs.wait(job, timeout=60)

    assert job.status == "done" and job.error is None
    assert job.summary == {"media": 1, "entities": 3, "failed": 1}
    assert job.progress["done"] == job.progress["total"] == 2
    types = [e["type"] for e in job.events]
    assert types[0] == "job_started" and types[-1] == "job_done" and types.count("job_done") == 1
    assert {"plan", "media_started", "media_created", "detections", "entity", "media_done"} <= set(types)
    assert types.count("progress") == 2 and types.count("entity") == 3
    assert any(e["type"] == "log" and e["level"] == "error" and "broken.jpg" in e["message"] for e in job.events)
    assert not good.path.exists() and not broken.path.exists()  # spooled uploads are removed

    (media_done,) = [e["media"] for e in job.events if e["type"] == "media_done"]
    by_label = sorted((e["label"], e["description"] or "") for e in media_done["entities"])
    assert by_label == [("knife", ""), ("person", "A person in a green jacket"), ("person", "A person in a red jacket")]
    persons = [e for e in media_done["entities"] if e["label"] == "person"]
    assert all(e["match_status"] == "new" and e["person"] for e in persons)
    assert len({e["person"]["id"] for e in persons}) == 2

    replay = [ev async for ev in jobs.subscribe(job)]
    assert replay == job.events  # late subscribers get the full history and the stream ends


async def test_same_person_is_matched_across_jobs(app_ctx: Any, tmp_path: Path) -> None:
    jobs: JobManager = app_ctx.jobs
    first = jobs.submit_analyze([_upload(tmp_path, "a.png", _scene(RED))], AnalyzeOptions())
    await jobs.wait(first)
    second = jobs.submit_analyze([_upload(tmp_path, "b.png", _scene(GREEN, RED))], AnalyzeOptions())
    await jobs.wait(second)
    entities = (await jobs._detail(second.media_ids[0]))["entities"]
    status = {e["description"]: e["match_status"] for e in entities}
    assert status == {"A person in a green jacket": "new", "A person in a red jacket": "matched"}


async def test_all_items_failing_marks_job_failed(app_ctx: Any, tmp_path: Path) -> None:
    job = app_ctx.jobs.submit_analyze([_upload(tmp_path, "x.jpg", b"junk")], AnalyzeOptions())
    await app_ctx.jobs.wait(job)
    assert job.status == "error" and "failed" in (job.error or "")
    assert job.events[-1]["type"] == "job_done" and job.events[-1]["job"]["status"] == "error"


async def test_cancel_running_job(app_ctx: Any, tmp_path: Path) -> None:
    jobs: JobManager = app_ctx.jobs
    jobs.media_concurrency = 1
    detector = app_ctx.registry.models["detector"]
    detector.gate = asyncio.Event()
    items = [_upload(tmp_path, f"{i}.png", _scene(RED)) for i in range(3)]
    job = jobs.submit_analyze(items, AnalyzeOptions())
    live = asyncio.create_task(_collect(jobs, job))
    await _wait_for(job, "media_created")
    jobs.cancel(job)
    detector.gate.set()
    await jobs.wait(job, timeout=30)

    assert job.status == "cancelled"
    assert job.media_ids == []  # the item in flight stops before saving, the rest never start
    assert sum(e["type"] == "media_started" for e in job.events) == 1
    events = await asyncio.wait_for(live, 5)
    assert events[-1]["type"] == "job_done" and events[-1]["job"]["status"] == "cancelled"
    assert all(not it.path.exists() for it in items)
    jobs.cancel(job)  # cancelling a finished job is a no-op
    assert job.status == "cancelled"


async def _collect(jobs: JobManager, job: Job) -> list[dict]:
    return [ev async for ev in jobs.subscribe(job)]


async def test_verification_rejects_false_positives(app_ctx: Any, tmp_path: Path) -> None:
    app_ctx.registry.models["vlm"] = FakeVLM(answer="no")
    job = app_ctx.jobs.submit_analyze([_upload(tmp_path, "k.png", _scene(BLUE, RED))], AnalyzeOptions(query="weapons", describe="none"))
    await app_ctx.jobs.wait(job)
    assert job.plan and job.plan["source"] == "lexicon"
    (knife,) = (await app_ctx.jobs._detail(job.media_ids[0]))["entities"]  # persons are not part of "weapons"
    assert knife["label"] == "knife" and knife["verified"] is False and knife["stage"] == "rejected"

    app_ctx.registry.models["vlm"] = FakeVLM(answer=True)
    job = app_ctx.jobs.submit_analyze([_upload(tmp_path, "k2.png", _scene(BLUE))], AnalyzeOptions(query="weapons", describe="none"))
    await app_ctx.jobs.wait(job)
    (knife,) = (await app_ctx.jobs._detail(job.media_ids[0]))["entities"]
    assert knife["verified"] is True and knife["concept"] == "weapons" and knife["verify_reason"] == "clearly visible"


async def test_scan_adds_only_new_detections(app_ctx: Any, tmp_path: Path) -> None:
    jobs: JobManager = app_ctx.jobs
    job = jobs.submit_analyze([_upload(tmp_path, "s.png", _scene(RED, BLUE))], AnalyzeOptions(query="people"))
    await jobs.wait(job)
    (mid,) = job.media_ids
    assert [e["label"] for e in (await jobs._detail(mid))["entities"]] == ["person"]

    scan = jobs.submit_scan(None, AnalyzeOptions(query="knives, people", describe="none"))
    await jobs.wait(scan)
    assert scan.status == "done" and scan.summary == {"scanned": 1, "media_with_hits": 1, "accepted": 2, "rejected": 0, "updated": 0}  # the new knife + the stored person
    assert sorted(e["label"] for e in (await jobs._detail(mid))["entities"]) == ["knife", "person"]


async def test_collect_paths(app_ctx: Any, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    for name in ("a.jpg", "sub/b.PNG", "sub/c.mp4", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    jobs: JobManager = app_ctx.jobs
    items = await jobs.collect_paths([str(tmp_path), str(tmp_path / "a.jpg"), str(tmp_path / "missing.jpg")], restricted=False)
    assert [(i.name, i.kind, i.temporary) for i in items] == [
        ("a.jpg", "image", False), ("b.PNG", "image", False), ("c.mp4", "video", False), ("a.jpg", "image", False),
    ]
    assert len(await jobs.collect_paths([str(tmp_path)], max_items=2, restricted=False)) == 2
