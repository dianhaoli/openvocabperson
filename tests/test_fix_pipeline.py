"""Pipeline behaviour: VLM reply parsing, concept attribution, library re-checks, cancellation, video tracking."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from PIL import Image

from ovp.pipeline.analyzer import AnalyzeOptions, Analyzer, Cancelled, parse_crop_reply
from ovp.pipeline.planner import QueryPlanner, may_involve_people
from ovp.types import Detection

from .fakes import BLUE, GREEN, RED, FakeLLM, FakeReID, FakeVLM, make_image


def _noop(*_: Any, **__: Any) -> None:
    return None


async def _analyze(ctx: Any, img: Image.Image, query: Optional[str], **opts: Any) -> tuple[str, list[Any]]:
    a: Analyzer = ctx.analyzer
    o = AnalyzeOptions(query=query, **opts)
    mid = await a.analyze_image(img, "x.png", o, await a.make_plan(o), _noop)
    return mid, list(await ctx.db.media_entities(mid))


async def _scan(ctx: Any, mid: str, query: str, **opts: Any) -> dict[str, int]:
    a: Analyzer = ctx.analyzer
    o = AnalyzeOptions(query=query, **opts)
    return await a.scan_media(mid, o, await a.make_plan(o), _noop)


# ── parsing ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "reply",
    [
        {"checks": {"q1": {"answer": True, "reason": "knife"}, "q2": "no"}},
        {"q1": {"answer": "yes", "reason": "knife"}, "q2": False},
        {"checks": [{"answer": True, "reason": "knife"}, False]},
    ],
)
def test_parse_crop_reply_dict_and_bare_answers(reply: dict) -> None:
    checks, _, _ = parse_crop_reply(json.dumps(reply), 2, describe=False)
    assert checks == [(True, "knife"), (False, "")]


def test_parse_crop_reply_attributes_without_summary_become_the_description() -> None:
    _, summary, attrs = parse_crop_reply(json.dumps({"q1": True, "clothing": "red shirt", "held_objects": "none", "actions": "walking"}), 1, describe=True)
    assert attrs == {"clothing": "red shirt", "actions": "walking"}  # the answer key is not an attribute
    assert summary == "clothing: red shirt; actions: walking"


# ── planner ───────────────────────────────────────────────────────────────────


async def test_llm_person_concepts_dropped_for_object_categories() -> None:
    reply = json.dumps({"concepts": [
        {"name": "dangerous things", "prompts": ["knife", "gun"], "verify": True, "question": "Is this dangerous?"},
        {"name": "dangerous people", "kind": "person_attribute", "prompts": ["person"], "question": "Is this person armed?"},
    ]})
    plan = await QueryPlanner().plan("dangerous things", llm=FakeLLM(reply))  # type: ignore[arg-type]
    assert [c.name for c in plan.concepts] == ["dangerous things"] and "person" not in plan.prompts
    plan = await QueryPlanner().plan("dangerous things or armed people", llm=FakeLLM(reply))  # type: ignore[arg-type]
    assert [c.kind for c in plan.concepts] == ["object", "person_attribute"]


def test_people_detection_in_queries() -> None:
    assert not may_involve_people("dangerous things") and not may_involve_people("weapons, sharp objects")
    assert may_involve_people("suspicious things") and may_involve_people("people fighting") and may_involve_people("fighting")
    tie, glasses = QueryPlanner().offline_plan("person wearing a tie, sunglasses").concepts
    assert tie.kind == "person_attribute" and glasses.kind == "object" and glasses.prompts == ["sunglasses"]


# ── concept attribution ───────────────────────────────────────────────────────


async def test_rejected_and_person_attribute_entities_keep_their_concept(app_ctx: Any) -> None:
    app_ctx.registry.models["vlm"] = FakeVLM(answer="no")
    _, (knife,) = await _analyze(app_ctx, make_image([(BLUE, (10, 10, 80, 80))]), "weapons", describe="none")
    assert knife["stage"] == "rejected" and knife["verified"] is False and knife["concept"] == "weapons"

    app_ctx.registry.models["vlm"] = FakeVLM(answer=True)
    _, (person,) = await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120))]), "menacing people", describe="none")
    assert person["concept"] == "menacing people" and person["verified"] is True


async def test_unknown_verification_keeps_candidate_with_checked_concept(app_ctx: Any) -> None:
    app_ctx.registry.models["vlm"] = FakeVLM(fail=True)
    _, (person,) = await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120))]), "menacing people", describe="none")
    assert person["stage"] == "detected" and person["verified"] is None and person["concept"] == "menacing people"


def test_query_plans_use_a_lower_default_confidence(app_ctx: Any) -> None:
    a: Analyzer = app_ctx.analyzer
    planner = QueryPlanner()
    assert a._conf(AnalyzeOptions(), planner.default_plan()) == 0.2
    assert a._conf(AnalyzeOptions(query="ties"), planner.offline_plan("ties")) == 0.15
    assert a._conf(AnalyzeOptions(query="ties", conf=0.4), planner.offline_plan("ties")) == 0.4


# ── library scans ─────────────────────────────────────────────────────────────


async def test_scan_reverifies_existing_entities(app_ctx: Any) -> None:
    mid, (person,) = await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120))]), None, describe="none")
    assert person["concept"] is None

    app_ctx.registry.models["vlm"] = FakeVLM(answer="no")
    assert await _scan(app_ctx, mid, "menacing people", describe="none") == {"accepted": 0, "rejected": 1, "updated": 0}
    (same,) = await app_ctx.db.media_entities(mid)
    assert same["stage"] == "detected" and same["concept"] is None  # a valid person, it just does not match

    app_ctx.registry.models["vlm"] = FakeVLM(answer=True)
    assert await _scan(app_ctx, mid, "menacing people", describe="none") == {"accepted": 1, "rejected": 0, "updated": 1}
    (flagged,) = await app_ctx.db.media_entities(mid)
    assert flagged["id"] == person["id"] and flagged["concept"] == "menacing people" and flagged["verified"] is True

    # a rejected row matched by a later query's plain concept comes back
    await app_ctx.db.update_entity(person["id"], stage="rejected", verified=False)
    assert await _scan(app_ctx, mid, "people", describe="none") == {"accepted": 1, "rejected": 0, "updated": 1}
    (back,) = await app_ctx.db.media_entities(mid)
    assert back["stage"] == "detected" and back["concept"] == "people" and back["verified"] is None


async def test_concurrent_scans_of_one_media_do_not_duplicate(app_ctx: Any) -> None:
    mid, _ = await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120)), (BLUE, (150, 20, 220, 90))]), "people", describe="none")
    results = await asyncio.gather(*(_scan(app_ctx, mid, "knives", describe="none") for _ in range(3)))
    assert sorted(r["updated"] for r in results) == [0, 0, 0] and sum(r["accepted"] for r in results) == 3
    assert sorted(e["label"] for e in await app_ctx.db.media_entities(mid)) == ["knife", "person"]


# ── cancellation ──────────────────────────────────────────────────────────────


class _SlowSegmenter:
    spec = "fake:sam"
    kind = "sam"

    async def segment(self, image: Image.Image, dets: list[Detection]) -> None:
        await asyncio.Event().wait()


async def test_cancelled_image_is_marked_cancelled_and_segmentation_stopped(app_ctx: Any) -> None:
    seg = app_ctx.registry.models["segmenter"] = _SlowSegmenter()
    app_ctx.registry.defaults["segmenter"] = seg.spec
    a: Analyzer = app_ctx.analyzer
    cancel = threading.Event()
    cancel.set()
    o = AnalyzeOptions(query="people", segment=True)
    with pytest.raises(Cancelled):
        await a.analyze_image(make_image([(RED, (10, 10, 80, 120))]), "c.png", o, await a.make_plan(o), _noop, cancel)
    assert asyncio.all_tasks() == {asyncio.current_task()}  # the segmentation task was not left running
    (row,) = await app_ctx.db.pool.fetch("SELECT status FROM media")
    assert row["status"] == "cancelled"


# ── video ─────────────────────────────────────────────────────────────────────


class _CountingReID(FakeReID):
    def __init__(self) -> None:
        self.embedded = 0

    async def embed(self, images: list[Image.Image]) -> np.ndarray:
        self.embedded += len(images)
        return await super().embed(images)


def _write_video(path: Path, frames: int = 24, fps: float = 4.0) -> None:
    import cv2

    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))  # type: ignore[attr-defined]
    for i in range(frames):
        img = make_image([(RED, (10 + 4 * i, 20, 70 + 4 * i, 140))] + ([(GREEN, (200, 40, 260, 160))] if i >= frames // 2 else []))
        out.write(np.asarray(img)[:, :, ::-1].copy())
    out.release()


async def test_video_tracks_with_any_detector_and_moves_upload(app_ctx: Any, tmp_path: Path) -> None:
    reid = app_ctx.registry.models["reid"] = _CountingReID()
    src = tmp_path / "clip.mp4"
    _write_video(src)
    a: Analyzer = app_ctx.analyzer
    o = AnalyzeOptions(query="people", describe="none", video_fps=4.0)
    mid = await a.analyze_video(src, "clip.mp4", o, await a.make_plan(o), _noop, move=True)
    assert not src.exists()  # moved into storage, not copied
    m = await app_ctx.db.get_media(mid)
    assert m["status"] == "done" and Path(m["path"]).exists()
    ents = await app_ctx.db.media_entities(mid)
    assert len(ents) == 2 and all(e["track_id"] is not None and e["trajectory"] for e in ents)
    red = next(e for e in ents if e["t_start"] == 0)
    assert len(red["trajectory"]) == 24  # one detection per sampled frame
    assert len(app_ctx.registry.models["detector"].calls) == 24  # one device call per frame
    # Re-ID crops are embedded once (track stitching), not again in the main stage
    assert reid.embedded == sum(min(8, -(-n // 3)) for n in (24, 12))
