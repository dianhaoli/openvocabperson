from __future__ import annotations

import numpy as np
import pytest

from ovp.services.identity import IdentityService, agglomerate, leader_clusters
from ovp.storage.db import Database

from .seed import SPACE, add_entities, add_media, centroid, entity_rows, person_ids, vec

A = vec(1, 0, 0, 0)
B = vec(0, 1, 0, 0)
C = vec(0, 0, 1, 0)


# ── clustering ────────────────────────────────────────────────────────────────


def _sorted(clusters: list[list[int]]) -> list[list[int]]:
    return sorted(sorted(c) for c in clusters)


def test_agglomerate_groups_similar_vectors() -> None:
    vecs = np.stack([A, vec(1, 0.1, 0, 0), B, vec(0.1, 1, 0, 0), C])
    assert _sorted(agglomerate(vecs, ["m1", "m2", "m3", "m4", "m5"], 0.8)) == [[0, 1], [2, 3], [4]]


def test_agglomerate_cannot_link_within_media() -> None:
    vecs = np.stack([A, A, A])
    # 0 and 2 share a media: whichever of them merges with 1 first, the other stays alone
    clusters = _sorted(agglomerate(vecs, ["m1", "m2", "m1"], 0.5))
    assert len(clusters) == 2 and [1] not in clusters
    assert all(not {0, 2} <= set(c) for c in clusters)


def test_agglomerate_edge_cases() -> None:
    assert agglomerate(np.zeros((0, 4)), [], 0.5) == []
    assert agglomerate(np.stack([A]), ["m1"], 0.5) == [[0]]
    assert _sorted(agglomerate(np.stack([A, B]), ["m1", "m2"], 0.5)) == [[0], [1]]


def test_leader_clusters_respects_cannot_link() -> None:
    vecs = np.stack([A, vec(1, 0.05, 0, 0), A, B])
    assert _sorted(leader_clusters(vecs, ["m1", "m2", "m2", "m3"], 0.8)) == [[0, 1], [2], [3]]


# ── assign_media ──────────────────────────────────────────────────────────────


@pytest.fixture
def identity(db: Database) -> IdentityService:
    return IdentityService(db)


async def _assign(identity: IdentityService, vecs: list[np.ndarray], kind: str = "image", **kw) -> tuple[list[str], dict]:
    mid = await add_media(identity.db, kind=kind)
    ids = await add_entities(identity.db, mid, vecs, spans=kw.pop("spans", None))
    return ids, await identity.assign_media(list(zip(ids, vecs)), SPACE, **kw)


async def test_assign_media_new_matched_pending(db: Database, identity: IdentityService) -> None:
    (a0,), res = await _assign(identity, [A])
    assert res[a0]["status"] == "new"
    pa = res[a0]["person_id"]

    (a1, b1), res = await _assign(identity, [vec(1, 0.1, 0, 0), B])
    assert res[a1]["status"] == "matched" and res[a1]["person_id"] == pa and res[a1]["score"] > 0.9
    assert res[b1]["status"] == "new" and res[b1]["person_id"] not in (None, pa)

    (c2,), res = await _assign(identity, [vec(1, 0, 1, 0)])  # cos 0.707 with A: between review and match
    assert res[c2] == {"status": "pending", "person_id": None, "suggested": pa, "score": pytest.approx(0.707, abs=0.01)}

    rows = await entity_rows(db, [a0, a1, c2])
    assert rows[a1]["person_id"] == pa and rows[a1]["match_status"] == "matched"
    assert rows[c2]["person_id"] is None and rows[c2]["suggested_person_id"] == pa
    person = await db.get_person(pa)
    assert person["sighting_count"] == 2 and person["media_count"] == 2
    assert float(centroid_cos := np.dot(await centroid(db, pa), vec(1, 0.05, 0, 0))) > 0.999, centroid_cos
    assert [r["id"] for r in await db.pending_entities()] == [c2]


async def test_assign_media_one_person_per_image(db: Database, identity: IdentityService) -> None:
    (a0,), res = await _assign(identity, [A])
    pa = res[a0]["person_id"]
    (x, y), res = await _assign(identity, [vec(1, 0.05, 0, 0), vec(1, 0.2, 0, 0)])
    assert res[x]["person_id"] == pa and res[x]["status"] == "matched"
    assert res[y]["status"] == "new" and res[y]["person_id"] != pa


async def test_assign_media_video_conflicts_are_temporal(db: Database, identity: IdentityService) -> None:
    (a0,), res = await _assign(identity, [A])
    pa = res[a0]["person_id"]
    spans = {"early": (0.0, 2.0), "late": (5.0, 9.0), "overlap": (1.0, 6.0)}

    def conflicts_for(ids: list[str], names: list[str]):
        span = dict(zip(ids, (spans[n] for n in names)))
        return lambda a, b: not (span[a][1] < span[b][0] or span[b][1] < span[a][0])

    mid = await add_media(db, kind="video")
    ids = await add_entities(db, mid, [A, vec(1, 0.05, 0, 0)])
    res = await identity.assign_media(list(zip(ids, [A, vec(1, 0.05, 0, 0)])), SPACE, conflicts=conflicts_for(ids, ["early", "late"]))
    assert {r["person_id"] for r in res.values()} == {pa}  # re-entering the scene is the same person

    mid = await add_media(db, kind="video")
    ids = await add_entities(db, mid, [A, vec(1, 0.05, 0, 0)])
    res = await identity.assign_media(list(zip(ids, [A, vec(1, 0.05, 0, 0)])), SPACE, conflicts=conflicts_for(ids, ["early", "overlap"]))
    assert sorted(r["status"] for r in res.values()) == ["matched", "new"]


async def test_assign_media_empty(identity: IdentityService) -> None:
    assert await identity.assign_media([], SPACE) == {}


# ── review / manual assignment ────────────────────────────────────────────────


async def test_review_actions(db: Database, identity: IdentityService) -> None:
    (a0,), res = await _assign(identity, [A])
    pa = res[a0]["person_id"]
    (p1, p2, p3), res = await _assign(identity, [vec(1, 0, 1, 0), vec(1, 0, 0, 1), vec(1, 1, 0, 0)])
    assert all(r["status"] == "pending" and r["suggested"] == pa for r in res.values())

    await identity.review(p1, "accept")
    await identity.review(p2, "new")
    await identity.review(p3, "reject")
    rows = await entity_rows(db, [p1, p2, p3])
    assert rows[p1]["person_id"] == pa and rows[p1]["match_status"] == "manual" and rows[p1]["suggested_person_id"] is None
    new_pid = rows[p2]["person_id"]
    assert new_pid not in (None, pa) and rows[p2]["match_status"] == "manual"
    assert np.dot(await centroid(db, new_pid), vec(1, 0, 0, 1)) > 0.999
    assert rows[p3]["person_id"] is None and rows[p3]["match_status"] is None and rows[p3]["suggested_person_id"] is None
    assert await db.pending_entities() == []
    assert (await db.get_person(pa))["sighting_count"] == 2

    with pytest.raises(ValueError):
        await identity.review(p3, "accept")  # nothing suggested any more
    with pytest.raises(ValueError):
        await identity.review(p3, "explode")
    with pytest.raises(KeyError):
        await identity.review("e_missing", "accept")


async def test_manual_assign_moves_between_clusters(db: Database, identity: IdentityService) -> None:
    (a0,), res = await _assign(identity, [A])
    pa = res[a0]["person_id"]
    (b0,), res = await _assign(identity, [B])
    pb = res[b0]["person_id"]
    await identity.assign(b0, pa)
    assert (await db.get_entity(b0))["person_id"] == pa
    assert pb not in await person_ids(db)  # emptied unnamed cluster is removed
    assert (await db.get_person(pa))["sighting_count"] == 2


# ── merge / delete ────────────────────────────────────────────────────────────


async def test_merge_persons(db: Database, identity: IdentityService) -> None:
    (a0,), res = await _assign(identity, [A])
    pa = res[a0]["person_id"]
    (b0,), res = await _assign(identity, [B])
    pb = res[b0]["person_id"]
    await db.update_person(pb, label="Bob", notes="seen at the gate", is_watchlist=True)

    assert not await db.merge_persons(pa, pa)
    assert not await db.merge_persons(pa, "p_missing")
    assert await db.merge_persons(pa, pb)
    keep = await db.get_person(pa)
    assert (keep["label"], keep["notes"], keep["is_watchlist"], keep["sighting_count"]) == ("Bob", "seen at the gate", True, 2)
    assert await person_ids(db) == {pa}
    assert (await db.get_entity(b0))["person_id"] == pa
    assert np.dot(await centroid(db, pa), vec(1, 1, 0, 0)) > 0.999


async def test_delete_media_recomputes_clusters(db: Database, identity: IdentityService) -> None:
    (a0,), res = await _assign(identity, [A])
    pa = res[a0]["person_id"]
    await db.update_person(pa, label="Alice")
    (a1, b1), res = await _assign(identity, [vec(1, 0.3, 0, 0), B])
    pb = res[b1]["person_id"]
    assert res[a1]["person_id"] == pa
    media_id = (await db.get_entity(a1))["media_id"]

    out = await db.delete_media(media_id)
    assert out is not None and set(out["persons"]) == {pa, pb}
    assert await person_ids(db) == {pa}  # unnamed cluster without sightings is gone
    alice = await db.get_person(pa)
    assert alice["sighting_count"] == 1 and alice["representative_entity_id"] == a0
    assert np.dot(await centroid(db, pa), A) > 0.999
    assert await db.get_media(media_id) is None
    assert await db.delete_media(media_id) is None

    # deleting the last sighting keeps the named person but clears the dangling representative
    await db.delete_media((await db.get_entity(a0))["media_id"])
    alice = await db.get_person(pa)
    assert alice["sighting_count"] == 0 and alice["representative_entity_id"] is None


async def test_delete_person_unlinks_sightings(db: Database, identity: IdentityService) -> None:
    (a0,), res = await _assign(identity, [A])
    pa = res[a0]["person_id"]
    assert await db.delete_person(pa)
    assert not await db.delete_person(pa)
    row = await db.get_entity(a0)
    assert row["person_id"] is None and row["match_status"] is None


# ── recluster ─────────────────────────────────────────────────────────────────


async def test_recluster_with_named_anchor(db: Database, identity: IdentityService) -> None:
    (a0,), res = await _assign(identity, [A])
    alice = res[a0]["person_id"]
    await db.update_person(alice, label="Alice")
    # one sighting per media, every one put in its own cluster (as a too-strict threshold would)
    sightings = {"x": vec(1, 0.02, 0, 0), "y": vec(1, 0.03, 0, 0), "z": B, "w": vec(0.1, 1, 0, 0)}
    ids = {}
    for name, v in sightings.items():
        mid = await add_media(db)
        (ids[name],) = await add_entities(db, mid, [v])
        await identity.assign(ids[name], None)
    old_clusters = await person_ids(db) - {alice}
    # media with two people that both resemble Alice: at most one of them can be her
    mid = await add_media(db)
    pair = await add_entities(db, mid, [vec(1, 0, 0.45, 0), vec(1, 0, -0.45, 0)])
    for eid in pair:
        await identity.assign(eid, None)
    for name in ("x", "y", "z", "w"):  # reset the manual flag set by assign()
        await db.update_entity(ids[name], match_status="new")
    for eid in pair:
        await db.update_entity(eid, match_status="new")

    out = await identity.recluster(SPACE, threshold=0.8)
    rows = await entity_rows(db, [a0, *ids.values(), *pair])
    assert rows[a0]["person_id"] == alice
    assert rows[ids["x"]]["person_id"] == alice and rows[ids["y"]]["person_id"] == alice
    zw = rows[ids["z"]]["person_id"]
    assert zw == rows[ids["w"]]["person_id"] and zw in old_clusters  # an existing cluster id is reused
    assert sum(rows[e]["person_id"] == alice for e in pair) == 1
    assert out["sightings"] == 7 and out["persons"] == len(await person_ids(db)) == 3
    assert (await db.get_person(alice))["sighting_count"] == 4


async def test_recluster_without_sightings(identity: IdentityService) -> None:
    assert await identity.recluster(SPACE) == {"persons": 0, "changed": 0, "sightings": 0}


async def test_space_with_mixed_dimensions(db: Database, identity: IdentityService) -> None:
    """A space keeps working when its model changes size (e.g. ``reid:clip`` after a CLIP switch)."""
    (old,), _ = await _assign(identity, [vec(1, 0, 0)])
    (new,), res = await _assign(identity, [A])
    assert res[new]["status"] == "new"
    assert [e for e, _ in await db.knn(SPACE, vec(1, 0, 0), 10, {})] == [old]
    assert [e for e, _ in await db.knn(SPACE, A, 10, {})] == [new]
    await identity.assign(old, res[new]["person_id"])
    assert np.dot(await centroid(db, res[new]["person_id"]), A) > 0.999
    out = await identity.recluster(SPACE)
    assert out["sightings"] == 1
