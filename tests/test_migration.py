"""Upgrade of a database written by the pre-``ovp`` code (``database.py`` at commit 746464a)."""

from __future__ import annotations

import importlib.util
import subprocess
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import AsyncIterator

import asyncpg
import numpy as np
import pytest

from ovp.storage.db import Database

from .conftest import drop_database, recreate_database

LEGACY_COMMIT = "746464a"
LEGACY_DB = "ovp_test_legacy"
REID_SPACE = "reid:osnet:osnet_x1_0"
REPO = Path(__file__).resolve().parent.parent


def load_legacy_module(tmp_path: Path) -> ModuleType:
    try:
        source = subprocess.run(["git", "show", f"{LEGACY_COMMIT}:database.py"], cwd=REPO, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"legacy schema source unavailable: {e}")
    path = tmp_path / "legacy_database.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("legacy_database", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
async def legacy_url(database: Database) -> AsyncIterator[str]:  # ``database``: skip without PostgreSQL
    yield await recreate_database(LEGACY_DB)
    await drop_database(LEGACY_DB)


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return v / np.linalg.norm(v)


async def write_legacy_data(legacy: ModuleType, url: str) -> dict[str, str]:
    old = legacy.Database(url)
    await old.connect()
    try:
        t = datetime(2024, 5, 1, 12, 0, 0)
        await old.create_session("s1", "/data/full/s1.jpg", 1920, 1080, created_at=t)
        await old.create_session("s2", "/data/full/s2.jpg", 640, 480)
        await old.create_entity("o1", "s1", "person", 0.91, (10, 20, 110, 320), "/data/crops/o1.jpg", "Man in a red jacket", "vlm_full", created_at=t)
        await old.create_entity("o2", "s1", "dog", 0.55, (200, 210, 260, 300), "/data/crops/o2.jpg", None, "yolo_only")
        await old.create_entity("o3", "s2", "person", 0.25, (5, 5, 50, 150), "/data/crops/o3.jpg", None, "low_confidence")
        await old.update_entity_reid_embedding("o1", unit(1))
        await old.update_entity_reid_embedding("o3", unit(2))
        pid = await old.create_person("Alice", True, unit(1), "o1", notes="watch her")
        await old.link_entity_to_person("o1", pid, 0.93, "matched")
        await old.link_entity_to_person("o3", pid, 0.8, "matched")
    finally:
        await old.close()
    return {"person": pid}


async def test_legacy_schema_is_migrated(tmp_path: Path, legacy_url: str) -> None:
    legacy = load_legacy_module(tmp_path)
    ids = await write_legacy_data(legacy, legacy_url)

    db = Database(legacy_url)
    await db.connect()
    try:
        media = {m["id"]: m for m in await db.list_media()}
        assert set(media) == {"s1", "s2"}
        assert (media["s1"]["kind"], media["s1"]["path"], media["s1"]["width"], media["s1"]["height"]) == ("image", "/data/full/s1.jpg", 1920, 1080)
        assert media["s1"]["entity_count"] == 2 and media["s1"]["person_count"] == 1

        ents = {e["id"]: e for e in await db.media_entities_many(["s1", "s2"])}
        assert set(ents) == {"o1", "o2", "o3"}
        o1 = ents["o1"]
        assert (o1["label"], o1["description"], o1["stage"], o1["crop_path"]) == ("person", "Man in a red jacket", "described", "/data/crops/o1.jpg")
        assert (o1["x1"], o1["y1"], o1["x2"], o1["y2"]) == (10, 20, 110, 320)
        assert (o1["person_id"], o1["person_label"], o1["match_status"]) == (ids["person"], "Alice", "matched")
        assert o1["match_score"] == pytest.approx(0.93)
        assert ents["o2"]["stage"] == "detected" and ents["o3"]["stage"] == "low_confidence"

        person = await db.get_person(ids["person"])
        assert (person["label"], person["is_watchlist"], person["notes"], person["space"]) == ("Alice", True, "watch her", REID_SPACE)
        assert person["sighting_count"] == 2 and person["representative_entity_id"] == "o1"
        assert np.allclose(await db.get_embedding("o1", REID_SPACE), unit(1), atol=1e-5)
        assert await db.get_embedding("o2", REID_SPACE) is None
        (match,) = await db.match_persons(REID_SPACE, unit(1))
        assert match[0] == ids["person"] and match[1] > 0.5

        hits = await db.fts("red jacket", 10, {})
        assert [h for h, _ in hits] == ["o1"]

        assert db.pool
        tables = {r["tablename"] for r in await db.pool.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        assert {"legacy_sessions", "legacy_entities", "legacy_persons"} <= tables
        assert await db.pool.fetchval("SELECT COUNT(*) FROM legacy_entities") == 3
    finally:
        await db.close()

    # a second start is a no-op: nothing is migrated twice
    db = Database(legacy_url)
    await db.connect()
    try:
        assert await db.count_media() == 2
        assert db.pool and await db.pool.fetchval("SELECT COUNT(*) FROM entities") == 3
    finally:
        await db.close()


async def test_legacy_persons_without_vector_column(tmp_path: Path, legacy_url: str) -> None:
    """Installs without pgvector kept the centroid only as JSON text (and no sighting embeddings)."""
    legacy = load_legacy_module(tmp_path)
    ids = await write_legacy_data(legacy, legacy_url)
    conn = await asyncpg.connect(legacy_url)
    try:
        await conn.execute("ALTER TABLE persons DROP COLUMN centroid")
        await conn.execute("ALTER TABLE entities DROP COLUMN reid_embedding_json, DROP COLUMN reid_embedding")
    finally:
        await conn.close()

    db = Database(legacy_url)
    await db.connect()
    try:
        assert (await db.get_person(ids["person"]))["label"] == "Alice"
        assert await db.get_embedding("o1", REID_SPACE) is None
        assert [p for p, _ in await db.match_persons(REID_SPACE, unit(1))] == [ids["person"]]
    finally:
        await db.close()
