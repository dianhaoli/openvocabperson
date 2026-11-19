"""Hybrid retrieval with reciprocal-rank fusion.

Signals (each optional, all run concurrently):
* ``fts``      – Postgres full-text over label/concept (weight A) and VLM description (weight B)
* ``label``    – label / concept match, with offline lexicon expansion ("weapons" → knife, gun ...)
* ``semantic`` – text-embedding similarity between the query and VLM descriptions
* ``clip``     – CLIP text→image similarity against crop embeddings
* ``image``    – CLIP image→image (query photo)
* ``reid``     – person Re-ID similarity (query photo containing a person)

Scores are fused with weighted RRF so signals with incomparable scales combine sensibly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import numpy as np
from PIL import Image

from .. import serialize
from ..models.registry import ModelRegistry
from ..pipeline.planner import QueryPlanner
from ..storage.db import Database, rows_by_id

log = logging.getLogger("ovp.search")

RRF_K = 60
WEIGHTS = {"fts": 1.0, "label": 1.2, "semantic": 1.0, "clip": 0.8, "image": 1.0, "reid": 1.2}
# Nearest neighbours below these cosine similarities are unrelated and only add noise to the fusion.
MIN_SIM = {"semantic": 0.2, "clip": 0.18, "image": 0.5, "reid": 0.3}


class SearchService:
    def __init__(self, db: Database, registry: ModelRegistry, planner: QueryPlanner):
        self.db, self.reg, self.planner = db, registry, planner
        self._text_cache: dict[tuple[str, str], np.ndarray] = {}

    async def _embed_query(self, model: Any, kind: str, text: str) -> Optional[np.ndarray]:
        key = (kind + getattr(model, "space", ""), text)
        if key in self._text_cache:
            return self._text_cache[key]
        try:
            v = (await model.embed_texts([text]))[0]
        except Exception as e:
            log.warning("query embedding (%s) failed: %s", kind, e)
            return None
        if len(self._text_cache) > 512:
            self._text_cache.clear()
        self._text_cache[key] = v
        return v

    async def search(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        filters = {k: v for k, v in (filters or {}).items() if v}
        pool = max(limit * 3, 50)
        tasks: dict[str, Any] = {}
        if text and text.strip():
            text = text.strip()
            tasks["fts"] = self.db.fts(text, pool, filters)
            plan = self.planner.offline_plan(text)
            terms = plan.prompts + [c.name for c in plan.concepts]
            tasks["label"] = self.db.label_match(list(dict.fromkeys(terms)), pool, filters)
            temb = await self.reg.ensure("text_embedder")
            if temb is not None:
                tasks["semantic"] = self._knn_text(temb, "semantic", text, pool, filters)
            clip = await self.reg.ensure("embedder")
            if clip is not None:
                tasks["clip"] = self._knn_text(clip, "clip", text, pool, filters)
        if image is not None:
            clip = await self.reg.ensure("embedder")
            if clip is not None:
                tasks["image"] = self._knn_image(clip, image, pool, filters)
            reid = await self.reg.ensure("reid")
            if reid is not None:
                tasks["reid"] = self._knn_reid(reid, image, pool, filters)
        if not tasks:
            return []
        names = list(tasks)
        outs = await asyncio.gather(*tasks.values(), return_exceptions=True)
        fused: dict[str, float] = {}
        raw: dict[str, dict[str, float]] = {}
        for name, res in zip(names, outs):
            if isinstance(res, Exception):
                log.warning("search signal %s failed: %s", name, res)
                continue
            for rank, (eid, s) in enumerate(res):
                fused[eid] = fused.get(eid, 0.0) + WEIGHTS[name] / (RRF_K + rank + 1)
                raw.setdefault(eid, {})[name] = round(float(s), 4)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"entity_id": eid, "score": round(sc, 5), "scores": raw[eid]} for eid, sc in ranked]

    async def _knn_text(self, model: Any, kind: str, text: str, k: int, filters: dict) -> list[tuple[str, float]]:
        v = await self._embed_query(model, kind, text)
        if v is None:
            return []
        return await self.db.knn(model.space, v, k, filters, min_sim=MIN_SIM[kind])

    async def _knn_image(self, clip: Any, image: Image.Image, k: int, filters: dict) -> list[tuple[str, float]]:
        v = (await clip.embed_images([image]))[0]
        return await self.db.knn(clip.space, v, k, filters, min_sim=MIN_SIM["image"])

    async def _knn_reid(self, reid: Any, image: Image.Image, k: int, filters: dict) -> list[tuple[str, float]]:
        crop = await self.person_crop(image)
        if crop is None:
            return []
        v = (await reid.embed([crop]))[0]
        return await self.db.knn(reid.space, v, k, {**filters, "labels": ["person"]}, min_sim=MIN_SIM["reid"])

    async def person_crop(self, image: Image.Image) -> Optional[Image.Image]:
        """Largest detected person, or the image itself if no detector / no person."""
        det = await self.reg.ensure("detector")
        if det is None:
            return image
        try:
            dets = [d for d in await det.detect(image, ["person"], 0.3) if d.label == "person"]
        except Exception:
            dets = []
        if not dets:
            return image
        d = max(dets, key=lambda x: x.area)
        return image.crop(tuple(int(v) for v in d.box))

    async def hydrate(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hits:
            return []
        ents = rows_by_id(await self.db.get_entities([h["entity_id"] for h in hits]))
        medias = rows_by_id(await self.db.get_media_many(list({e["media_id"] for e in ents.values()})))
        out = []
        for h in hits:
            e = ents.get(h["entity_id"])
            if e is None:
                continue
            m = medias.get(e["media_id"])
            out.append({"entity": serialize.entity(e), "media": serialize.media(m) if m else None, "score": h["score"], "scores": h["scores"]})
        return out
