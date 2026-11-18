"""Person identity: matching new sightings to clusters, review queue, and re-clustering.

Fixes vs. the original implementation
* matching is done per media item with a uniqueness constraint (two people in the same photo /
  video can never be linked to the same person)
* a process-wide lock makes concurrent uploads unable to create duplicate clusters
* "pending" matches keep a ``suggested_person_id`` and can be accepted/rejected
* centroids are recomputed with pgvector ``AVG`` whenever membership changes (incl. deletes)
* thresholds depend on the Re-ID backend (OSNet / CLIP / colour histogram)
* ``recluster`` performs constrained average-linkage clustering over all sightings
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from typing import Any, Callable, Optional

import numpy as np

from ..config import get_settings
from ..models.embedders import l2n
from ..storage.db import Database

log = logging.getLogger("ovp.identity")


def reid_kind_of(space: str) -> str:
    # "reid:osnet:osnet_x1_0" → "osnet"
    parts = space.split(":")
    return parts[1] if len(parts) > 1 else parts[0]


def agglomerate(vecs: np.ndarray, groups: list[str], threshold: float) -> list[list[int]]:
    """Centroid-linkage agglomerative clustering with cannot-link on equal ``groups`` (media ids)."""
    n = len(vecs)
    if n == 0:
        return []
    if n > 4000:  # leader clustering for very large sets
        clusters: list[list[int]] = []
        cents: list[np.ndarray] = []
        medias: list[set[str]] = []
        for i in range(n):
            best, bs = -1, threshold
            for k, c in enumerate(cents):
                s = float(vecs[i] @ c)
                if s >= bs and groups[i] not in medias[k]:
                    best, bs = k, s
            if best < 0:
                clusters.append([i]); cents.append(vecs[i].copy()); medias.append({groups[i]})
            else:
                clusters[best].append(i); medias[best].add(groups[i])
                cents[best] = l2n(vecs[clusters[best]].sum(0))
        return clusters
    X = l2n(vecs.astype(np.float32))
    sums = X.copy()
    cents = X.copy()
    members: list[Optional[list[int]]] = [[i] for i in range(n)]
    medias = [{g} for g in groups]
    S = cents @ cents.T
    np.fill_diagonal(S, -np.inf)
    g = np.array(groups, dtype=object)
    S[g[:, None] == g[None, :]] = -np.inf  # cannot-link inside the same media
    while True:
        flat = int(np.argmax(S))
        i, j = divmod(flat, n)
        if not np.isfinite(S[i, j]) or S[i, j] < threshold:
            break
        if medias[i] & medias[j]:
            S[i, j] = S[j, i] = -np.inf
            continue
        # merge j into i
        members[i] = members[i] + members[j]  # type: ignore[operator]
        members[j] = None
        medias[i] |= medias[j]
        sums[i] += sums[j]
        cents[i] = l2n(sums[i])
        S[j, :] = -np.inf
        S[:, j] = -np.inf
        row = cents @ cents[i]
        alive = np.array([m is not None for m in members])
        row[~alive] = -np.inf
        row[i] = -np.inf
        for k in np.nonzero(alive)[0]:
            if medias[i] & medias[k]:
                row[k] = -np.inf
        S[i, :] = row
        S[:, i] = row
    return [m for m in members if m]


class IdentityService:
    def __init__(self, db: Database):
        self.db = db
        self.lock = asyncio.Lock()

    def thresholds(self, space: str):
        return get_settings().thresholds_for(reid_kind_of(space))

    async def assign_media(
        self,
        items: list[tuple[str, np.ndarray]],
        space: str,
        conflicts: Optional[Callable[[str, str], bool]] = None,
    ) -> dict[str, dict[str, Any]]:
        """Link the person sightings of ONE media item. Returns entity_id → result.

        ``conflicts(a, b)`` says whether two sightings cannot be the same person; by default all
        sightings in one image conflict, for video only temporally overlapping tracks do.
        """
        conflicts = conflicts or (lambda a, b: True)
        if not items:
            return {}
        th = self.thresholds(space)
        results: dict[str, dict[str, Any]] = {}
        async with self.lock:
            cands: list[tuple[float, str, str]] = []
            best: dict[str, tuple[Optional[str], float]] = {}
            for eid, vec in items:
                matches = await self.db.match_persons(space, vec, k=8)
                best[eid] = matches[0] if matches else (None, -1.0)
                cands += [(s, eid, pid) for pid, s in matches]
            cands.sort(reverse=True)
            owners: dict[str, list[str]] = {}  # person → sightings of this media linked to it
            done: set[str] = set()

            def free(pid: str, eid: str) -> bool:
                return all(not conflicts(eid, o) for o in owners.get(pid, []))

            for s, eid, pid in cands:
                if s < th.match:
                    break
                if eid in done or not free(pid, eid):
                    continue
                done.add(eid)
                owners.setdefault(pid, []).append(eid)
                await self.db.link_entity(eid, pid, "matched", s)
                results[eid] = {"status": "matched", "person_id": pid, "score": s}
            for eid, vec in items:
                if eid in done:
                    continue
                # best remaining candidate not claimed by someone else in this media
                alt = [(s, pid) for s, e, pid in cands if e == eid and free(pid, eid)]
                s, pid = max(alt) if alt else (-1.0, None)
                if pid is not None and s >= th.review:
                    await self.db.link_entity(eid, None, "pending", s, suggested=pid)
                    results[eid] = {"status": "pending", "person_id": None, "suggested": pid, "score": s}
                else:
                    new_pid = await self.db.create_person(space, vec, eid)
                    owners.setdefault(new_pid, []).append(eid)
                    await self.db.link_entity(eid, new_pid, "new", s if s > -1 else None)
                    results[eid] = {"status": "new", "person_id": new_pid, "score": s if s > -1 else None}
        return results

    async def assign(self, entity_id: str, person_id: Optional[str]) -> None:
        async with self.lock:
            if person_id is None:
                space = await self._entity_space(entity_id)
                vec = await self.db.get_embedding(entity_id, space) if space else None
                person_id = await self.db.create_person(space or "manual", vec, entity_id)
            await self.db.link_entity(entity_id, person_id, "manual", None)

    async def review(self, entity_id: str, action: str) -> None:
        row = await self.db.get_entity(entity_id)
        if row is None:
            raise KeyError(entity_id)
        if action == "accept":
            if not row["suggested_person_id"]:
                raise ValueError("no suggested person to accept")
            await self.assign(entity_id, row["suggested_person_id"])
        elif action == "new":
            await self.assign(entity_id, None)
        elif action == "reject":
            await self.db.link_entity(entity_id, None, None, None)
        else:
            raise ValueError(f"unknown action {action!r}")

    async def _entity_space(self, entity_id: str) -> Optional[str]:
        assert self.db.pool
        return await self.db.pool.fetchval("SELECT space FROM embeddings WHERE entity_id = $1 AND space LIKE 'reid:%' LIMIT 1", entity_id)

    async def search_by_vector(self, space: str, vec: np.ndarray, limit: int = 15) -> list[tuple[str, float]]:
        return await self.db.match_persons(space, vec, k=limit)

    async def recluster(self, space: str, threshold: Optional[float] = None) -> dict[str, int]:
        """Re-cluster every person sighting in ``space``.

        Named / watch-listed / manually confirmed identities are anchors and keep their sightings;
        everything else is clustered from scratch, then clusters are attached to the closest
        anchor when similar enough, re-using existing cluster ids where possible.
        """
        th = self.thresholds(space)
        thr = threshold if threshold is not None else th.match
        async with self.lock:
            rows = await self.db.get_embeddings(space, only_persons=True)
            if not rows:
                return {"persons": 0, "changed": 0, "sightings": 0}
            persons = {r["id"]: r for r in (await self.db.list_persons(limit=100000))[0]}
            named = {pid for pid, p in persons.items() if p["label"] or p["is_watchlist"] or p["notes"]}
            fixed = [(eid, v, m, pid) for eid, v, m, pid, st in rows if pid in named or st == "manual"]
            free = [(eid, v, m, pid) for eid, v, m, pid, st in rows if not (pid in named or st == "manual")]
            anchors: dict[str, np.ndarray] = {}
            anchor_media: dict[str, set[str]] = {}
            for eid, v, m, pid in fixed:
                if pid:
                    anchors[pid] = anchors.get(pid, 0) + v  # type: ignore[assignment]
                    anchor_media.setdefault(pid, set()).add(m)
            anchors = {k: l2n(v) for k, v in anchors.items()}
            vecs = np.stack([v for _, v, _, _ in free]) if free else np.zeros((0, 1))
            clusters = await asyncio.to_thread(agglomerate, vecs, [m for _, _, m, _ in free], thr) if free else []
            assignments: dict[str, Optional[str]] = {}
            new_clusters: list[list[str]] = []
            claimed: set[str] = set()
            for cl in sorted(clusters, key=len, reverse=True):
                ids = [free[i][0] for i in cl]
                cmedia = {free[i][2] for i in cl}
                cent = l2n(vecs[cl].sum(0))
                target = None
                if anchors:
                    sims = sorted(((float(cent @ a), pid) for pid, a in anchors.items() if not (anchor_media.get(pid, set()) & cmedia)), reverse=True)
                    if sims and sims[0][0] >= thr:
                        target = sims[0][1]
                if target is None:
                    prev = Counter(free[i][3] for i in cl if free[i][3] and free[i][3] not in named)
                    for pid, _ in prev.most_common():
                        if pid not in claimed:
                            target = pid
                            break
                if target is None:
                    new_clusters.append(ids)
                    continue
                claimed.add(target)
                for e in ids:
                    assignments[e] = target
            changed = await self.db.apply_clustering(assignments, new_clusters, space)
            total = len({*assignments.values(), *named}) + len(new_clusters)
            return {"persons": total, "changed": changed, "sightings": len(rows)}
