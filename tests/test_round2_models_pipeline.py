"""Round 2, models and pipeline: weights cannot come from stored media, crop quality at the frame
border, scan re-check semantics, model specs resolved before media is stored, and moving person
sightings of an older Re-ID space into the active one."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from PIL import Image, ImageOps

from ovp.config import REID_THRESHOLDS, Settings
from ovp.models.embedders import ColorReID, l2n
from ovp.models.registry import InvalidSpec, ModelRegistry, normalize_spec
from ovp.pipeline.analyzer import AnalyzeOptions, Analyzer, _crop_quality
from ovp.services.identity import reembed_orphaned
from ovp.storage.files import FileStore, safe_suffix
from ovp.types import Detection

from .fakes import RED, FakeReID, FakeVLM, make_image


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


# ── stored file names and weights ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "kind", "suffix"),
    [("clip.MOV", "video", ".mov"), ("x.pt", "video", ".mp4"), ("noext", "video", ".mp4"), ("a.png", "image", ".jpg"), ("evil.pt", "image", ".jpg")],
)
def test_stored_suffix_is_never_the_clients_choice(name: str, kind: str, suffix: str) -> None:
    assert safe_suffix(name, kind) == suffix


async def test_import_file_uses_a_video_suffix(tmp_path: Path) -> None:
    store = FileStore(tmp_path / "storage")
    src = tmp_path / "upload.pt"
    src.write_bytes(b"not really a video")
    stored = await store.import_file("m_x", src, move=True)
    assert stored.name == "m_x.mp4" and not src.exists()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.weights_dir = tmp_path / "weights"
    s.storage_dir = s.weights_dir / "storage"  # the worst case: storage inside an allowed folder
    (s.storage_dir / "media").mkdir(parents=True)
    return s


def test_weights_never_load_from_media_storage(settings: Settings) -> None:
    upload = settings.storage_dir / "media" / "m_abc.pt"
    upload.write_bytes(b"pickle")
    (settings.weights_dir / "link.pt").symlink_to(upload)
    (settings.weights_dir / "linkdir").symlink_to(settings.storage_dir, target_is_directory=True)
    for spec in (f"yolo:{upload}", f"yolo:{settings.weights_dir / 'link.pt'}", "yolo:link.pt", f"qwen:{settings.weights_dir / 'linkdir'}", f"st:{settings.storage_dir}"):
        role = "vlm" if spec.startswith("qwen") else "text_embedder" if spec.startswith("st") else "detector"
        with pytest.raises(InvalidSpec, match="media storage"):
            normalize_spec(role, spec, settings)


def test_weights_need_a_known_suffix(settings: Settings) -> None:
    (settings.weights_dir / "clip.mp4").write_bytes(b"x")
    (settings.weights_dir / "notes.txt").write_text("x")
    (settings.weights_dir / "qwen-local").mkdir()
    with pytest.raises(InvalidSpec):
        normalize_spec("detector", f"yolo:{settings.weights_dir / 'clip.mp4'}", settings)
    with pytest.raises(InvalidSpec):
        normalize_spec("detector", "yolo:yolo11n.yaml", settings)
    with pytest.raises(InvalidSpec):
        normalize_spec("text_embedder", f"st:{settings.weights_dir / 'notes.txt'}", settings)
    for suffix in (".pt", ".pth", ".ts", ".onnx", ".engine", ".safetensors", ".bin"):
        assert normalize_spec("detector", f"yolo:{settings.weights_dir / 'sub' / ('m' + suffix)}", settings).endswith(suffix)
    local = settings.weights_dir / "qwen-local"
    assert normalize_spec("vlm", f"qwen:{local}", settings) == "qwen:qwen-local"  # a model folder
    assert normalize_spec("vlm", "qwen:Qwen/Qwen2.5-VL-3B-Instruct", settings) == "qwen:Qwen/Qwen2.5-VL-3B-Instruct"


def test_video_capability_does_not_depend_on_the_detector_kind(settings: Settings) -> None:
    settings.detector = "owlv2:google/owlv2-base-patch16-ensemble"
    reg = ModelRegistry(settings)
    assert reg.capabilities()["video"] is True


ASSETS = Path("/usr/local/lib/python3.13/site-packages/ultralytics/assets")
PEOPLE = {"zidane.jpg": [(750, 43, 1148, 716), (128, 203, 1106, 716)], "bus.jpg": [(672, 388, 810, 878), (53, 400, 244, 900), (226, 407, 342, 855)]}


@pytest.mark.skipif(not ASSETS.exists(), reason="ultralytics sample images not installed")
def test_color_reid_thresholds_fit_the_descriptor() -> None:
    """Re-encoded / rescaled / mirrored crops of one person match; different people do not."""
    th = REID_THRESHOLDS["color"]
    crops = [Image.open(ASSETS / name).convert("RGB").crop(b) for name, boxes in PEOPLE.items() for b in boxes]
    vecs = [ColorReID.describe(c) for c in crops]
    for c, v in zip(crops, vecs):
        small = c.resize((c.width // 3, c.height // 3))
        for variant in (small, ImageOps.mirror(c), c.crop((c.width // 20, 0, c.width - c.width // 20, c.height))):
            assert float(v @ ColorReID.describe(variant)) >= th.match + 0.05
    different = [float(vecs[i] @ vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    assert max(different) <= th.match - 0.05 and th.review < th.match


# ── video crop quality ────────────────────────────────────────────────────────


def test_border_clipped_boxes_make_worse_crops() -> None:
    w, h = 640, 480
    inside = Detection(box=(100, 100, 200, 300), score=0.8, label="person")
    clipped = Detection(box=(0, 100, 130, 330), score=0.8, label="person")  # larger, but cut by the left edge
    corner = Detection(box=(0, 0, 140, 340), score=0.8, label="person")
    assert clipped.area > inside.area
    assert _crop_quality(inside, w, h) > _crop_quality(clipped, w, h) > _crop_quality(corner, w, h)


# ── scan re-checks ────────────────────────────────────────────────────────────


async def test_plain_concept_never_replaces_a_verified_one(app_ctx: Any) -> None:
    mid, (person,) = await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120))]), "menacing people", describe="none")
    assert person["verified"] is True and person["concept"] == "menacing people"
    assert await _scan(app_ctx, mid, "people", describe="none") == {"accepted": 1, "rejected": 0, "updated": 0}
    (same,) = await app_ctx.db.media_entities(mid)
    assert (same["concept"], same["verified"], same["verify_reason"]) == ("menacing people", True, person["verify_reason"])


async def test_plain_relabel_clears_the_old_concepts_verdict(app_ctx: Any) -> None:
    mid, (person,) = await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120))]), "people", describe="none")
    await app_ctx.db.update_entity(person["id"], concept="old concept", verified=False, verify_reason="not it")
    assert await _scan(app_ctx, mid, "people", describe="none") == {"accepted": 1, "rejected": 0, "updated": 1}
    (same,) = await app_ctx.db.media_entities(mid)
    assert (same["concept"], same["verified"], same["verify_reason"]) == ("people", None, None)


async def test_denied_reverification_of_the_stored_concept_is_stored(app_ctx: Any) -> None:
    mid, (person,) = await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120))]), "menacing people", describe="none")
    assert person["verified"] is True
    app_ctx.registry.models["vlm"] = FakeVLM(answer="no")
    assert await _scan(app_ctx, mid, "menacing people", describe="none") == {"accepted": 0, "rejected": 1, "updated": 1}
    (same,) = await app_ctx.db.media_entities(mid)
    assert (same["id"], same["stage"], same["verified"], same["concept"]) == (person["id"], "rejected", False, "menacing people")
    assert same["verify_reason"] == "clearly visible"
    assert (await app_ctx.db.get_media(mid))["entity_count"] == 0  # the library agrees with the summary
    # denied again: nothing left to change
    assert await _scan(app_ctx, mid, "menacing people", describe="none") == {"accepted": 0, "rejected": 1, "updated": 0}


async def test_rejected_recheck_unlinks_the_person_and_acceptance_relinks(app_ctx: Any) -> None:
    db = app_ctx.db
    mid, (person,) = await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120))]), "menacing people", describe="none")
    pid = person["person_id"]
    assert pid and (await db.get_person(pid))["sighting_count"] == 1
    app_ctx.registry.models["vlm"] = FakeVLM(answer="no")
    await _scan(app_ctx, mid, "menacing people", describe="none")
    (same,) = await db.media_entities(mid)
    assert (same["stage"], same["person_id"], same["match_status"]) == ("rejected", None, None)
    assert await db.get_person(pid) is None  # the emptied, unnamed cluster is gone
    app_ctx.registry.models["vlm"] = FakeVLM(answer="yes")
    assert await _scan(app_ctx, mid, "menacing people", describe="none") == {"accepted": 1, "rejected": 0, "updated": 1}
    (again,) = await db.media_entities(mid)
    assert again["stage"] != "rejected" and again["person_id"] and again["match_status"] == "new"
    assert (await db.get_person(again["person_id"]))["sighting_count"] == 1


async def test_accepting_a_never_linked_rejection_embeds_and_links_it(app_ctx: Any) -> None:
    db = app_ctx.db
    app_ctx.registry.models["vlm"] = FakeVLM(answer="no")
    mid, (person,) = await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120))]), "menacing people", describe="none")
    reid = app_ctx.registry.models["reid"]
    assert person["stage"] == "rejected" and await db.get_embedding(person["id"], reid.space) is None
    app_ctx.registry.models["vlm"] = FakeVLM(answer="yes")
    await _scan(app_ctx, mid, "menacing people", describe="none")
    (same,) = await db.media_entities(mid)
    assert same["person_id"] and await db.get_embedding(same["id"], reid.space) is not None


# ── model specs before media ──────────────────────────────────────────────────


async def test_invalid_model_spec_creates_no_media(app_ctx: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    reg = app_ctx.registry
    ensure = reg.ensure

    async def validating_ensure(role: str, spec: Optional[str] = None) -> Any:
        reg.resolve(role, spec)  # like ModelRegistry.get: InvalidSpec before anything is built
        return await ensure(role, spec)

    monkeypatch.setattr(reg, "ensure", validating_ensure)
    files_before = sorted(os.listdir(app_ctx.store.media_dir))
    with pytest.raises(InvalidSpec):
        await _analyze(app_ctx, make_image([(RED, (10, 10, 80, 120))]), "people", detector="yolo:/etc/x.pt")
    assert await app_ctx.db.pool.fetchval("SELECT COUNT(*) FROM media") == 0
    assert sorted(os.listdir(app_ctx.store.media_dir)) == files_before


# ── Re-ID space migration ─────────────────────────────────────────────────────


class _OldReID(FakeReID):
    space = "reid:color"


async def test_reembed_orphaned_moves_sightings_and_persons(app_ctx: Any) -> None:
    db, store = app_ctx.db, app_ctx.store
    img = make_image([(RED, (10, 10, 80, 120))])
    app_ctx.registry.models["reid"] = _OldReID()
    _, (old,) = await _analyze(app_ctx, img, "people", describe="none")
    _, (gone,) = await _analyze(app_ctx, make_image([(RED, (150, 10, 220, 120))]), "people", describe="none")
    Path(gone["crop_path"]).unlink()
    pid = old["person_id"]
    assert pid and (await db.get_person(pid))["space"] == "reid:color"

    reid = FakeReID()
    app_ctx.registry.models["reid"] = reid
    res = await reembed_orphaned(db, store, reid)
    assert res == {"sightings": 1, "missing": 1, "persons": 1}
    person = await db.get_person(pid)
    assert person["space"] == reid.space
    vec = await db.get_embedding(old["id"], reid.space)
    expected = (await reid.embed([store.load_image(old["crop_path"])]))[0]
    assert vec is not None and float(l2n(np.asarray(vec)) @ expected) > 0.999
    assert await db.get_embedding(old["id"], "reid:color") is not None  # the old vector is kept

    assert await reembed_orphaned(db, store, reid) == {"sightings": 0, "missing": 1, "persons": 0}
    # the known person matches again in the new space
    _, (again,) = await _analyze(app_ctx, img, "people", describe="none")
    assert again["person_id"] == pid and again["match_status"] == "matched"
