"""Helpers that write media / entities / embeddings straight into the database."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ovp.models.embedders import l2n
from ovp.storage.db import Database, new_id

SPACE = "reid:osnet:fake"


def vec(*xs: float) -> np.ndarray:
    return l2n(np.asarray(xs, dtype=np.float32))


async def add_media(db: Database, kind: str = "image", name: Optional[str] = None, path: str = "/nonexistent/file.jpg") -> str:
    mid = new_id("m")
    await db.insert_media(id=mid, kind=kind, name=name or f"{mid}.jpg", path=path, width=640, height=480, status="done")
    return mid


async def add_entities(
    db: Database,
    media_id: str,
    vecs: Sequence[Optional[np.ndarray]] = (),
    *,
    label: str = "person",
    space: str = SPACE,
    spans: Optional[Sequence[tuple[float, float]]] = None,
    descriptions: Optional[Sequence[Optional[str]]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> list[str]:
    """One entity per entry of ``vecs`` (``None`` = no embedding); returns the entity ids."""
    rows, embs = [], []
    for i, v in enumerate(vecs):
        eid = new_id("e")
        t0, t1 = spans[i] if spans else (None, None)
        rows.append({
            "id": eid, "media_id": media_id, "label": label, "score": 0.9 - i * 0.01,
            "x1": 10.0 * i, "y1": 0.0, "x2": 10.0 * i + 50 + i, "y2": 100.0, "stage": "detected",
            "t_start": t0, "t_end": t1, "description": descriptions[i] if descriptions else None, **(extra or {}),
        })
        if v is not None:
            embs.append((eid, space, v))
    await db.insert_entities(rows, embs)
    return [r["id"] for r in rows]


async def entity_rows(db: Database, ids: Sequence[str]) -> dict[str, Any]:
    return {r["id"]: r for r in await db.get_entities(ids)}


async def person_ids(db: Database) -> set[str]:
    assert db.pool
    return {r["id"] for r in await db.pool.fetch("SELECT id FROM persons")}


async def centroid(db: Database, person_id: str) -> np.ndarray:
    """Unit-length centroid (cosine matching ignores its scale)."""
    assert db.pool
    text = await db.pool.fetchval("SELECT centroid::text FROM persons WHERE id = $1", person_id)
    return l2n(np.asarray([float(x) for x in text.strip("[]").split(",")], dtype=np.float32))
