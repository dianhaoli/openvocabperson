from __future__ import annotations

import pytest

from ovp.pipeline.planner import QueryPlanner
from ovp.services.search import RRF_K, WEIGHTS, SearchService
from ovp.storage.db import Database

from .fakes import RED, FakeDetector, FakeRegistry, FakeReID, FakeTextEmbedder, make_image
from .seed import SPACE, add_entities, add_media, vec

TEXT = FakeTextEmbedder()


async def _described(db: Database, media_id: str, label: str, description: str | None, **kw) -> str:
    (eid,) = await add_entities(db, media_id, [None], label=label, descriptions=[description], **kw)
    if description:
        await db.upsert_embeddings([(eid, TEXT.space, (await TEXT.embed_texts([f"{label}. {description}"]))[0])])
    return eid


@pytest.fixture
async def library(db: Database) -> dict[str, str]:
    img, vid = await add_media(db, kind="image"), await add_media(db, kind="video", name="clip.mp4")
    return {
        "img": img,
        "vid": vid,
        "red": await _described(db, img, "person", "A man in a red jacket holding a phone"),
        "blue": await _described(db, img, "person", "A woman in a blue dress"),
        "knife": await _described(db, img, "knife", None),
        "dog": await _described(db, vid, "dog", "A brown dog running"),
        "rejected": await _described(db, img, "person", "A man in a red jacket", extra={"stage": "rejected"}),
    }


@pytest.fixture
def search(db: Database) -> SearchService:
    return SearchService(db, FakeRegistry(text_embedder=TEXT), QueryPlanner())  # type: ignore[arg-type]


async def test_text_search_fuses_signals(search: SearchService, library: dict[str, str]) -> None:
    hits = await search.search("red jacket")
    assert hits[0]["entity_id"] == library["red"]
    assert {"fts", "semantic"} <= set(hits[0]["scores"])
    assert library["rejected"] not in {h["entity_id"] for h in hits}
    # an entity found by several signals at rank 0 gets the sum of their RRF contributions
    expected = sum(WEIGHTS[s] / (RRF_K + 1) for s in hits[0]["scores"])
    assert hits[0]["score"] == pytest.approx(expected, abs=1e-5)
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


async def test_label_signal_uses_lexicon_expansion(search: SearchService, library: dict[str, str]) -> None:
    hits = await search.search("weapons")
    assert hits[0]["entity_id"] == library["knife"] and "label" in hits[0]["scores"]


async def test_filters(search: SearchService, library: dict[str, str], db: Database) -> None:
    ids = lambda hits: {h["entity_id"] for h in hits}  # noqa: E731
    assert ids(await search.search("dog", filters={"kind": "video"})) == {library["dog"]}
    assert ids(await search.search("dog", filters={"kind": "image"})) == set()
    assert ids(await search.search("person", filters={"labels": ["PERSON"]})) == {library["red"], library["blue"]}
    assert ids(await search.search("dog", filters={"media_id": library["img"]})) == set()
    pid = await db.create_person(SPACE, None, library["blue"])
    await db.link_entity(library["blue"], pid, "manual", None)
    assert ids(await search.search("person", filters={"person_id": pid})) == {library["blue"]}
    assert ids(await search.search("woman", filters={"media_id": None, "labels": []})) >= {library["blue"]}  # empty filters ignored


async def test_limit_and_empty_queries(search: SearchService, library: dict[str, str]) -> None:
    assert len(await search.search("a person or a dog", limit=2)) == 2
    assert await search.search(None) == []
    assert await search.search("   ") == []


@pytest.mark.parametrize("text", ["the", "c++ & | ! ( :*", "don't <-> 'quoted'", "日本語", "%_\\"])
async def test_fts_and_label_queries_tolerate_odd_text(db: Database, library: dict[str, str], text: str) -> None:
    await db.fts(text, 10, {})
    await db.label_match([text], 10, {})


async def test_image_search_uses_reid(db: Database) -> None:
    mid = await add_media(db)
    red_like, other = await add_entities(db, mid, [vec(*RED), vec(30, 30, 220)])
    search = SearchService(db, FakeRegistry(detector=FakeDetector(), reid=FakeReID()), QueryPlanner())  # type: ignore[arg-type]
    hits = await search.search(image=make_image([(RED, (50, 40, 110, 200))]))
    assert [h["entity_id"] for h in hits] == [red_like]
    assert hits[0]["scores"]["reid"] > 0.99


async def test_hydrate_keeps_order_and_skips_missing(search: SearchService, library: dict[str, str]) -> None:
    hits = [{"entity_id": library["dog"], "score": 0.2, "scores": {}}, {"entity_id": "e_gone", "score": 0.1, "scores": {}},
            {"entity_id": library["red"], "score": 0.1, "scores": {"fts": 0.5}}]
    out = await search.hydrate(hits)
    assert [r["entity"]["id"] for r in out] == [library["dog"], library["red"]]
    assert out[0]["media"]["kind"] == "video" and out[1]["scores"] == {"fts": 0.5}
    assert await search.hydrate([]) == []
