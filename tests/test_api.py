from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
import pytest

from ovp.api.app import app

from .fakes import BLUE, GREEN, RED, make_image, png_bytes
from .seed import add_entities, add_media, vec


@pytest.fixture
async def client(app_ctx: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _scene(*colors: tuple[int, int, int]) -> bytes:
    return png_bytes(make_image([(c, (20 + 90 * i, 30, 90 + 90 * i, 200)) for i, c in enumerate(colors)]))


async def _analyze(client: httpx.AsyncClient, *scenes: bytes, options: dict | None = None) -> dict:
    files = [("files", (f"img{i}.png", data, "image/png")) for i, data in enumerate(scenes)]
    r = await client.post("/api/analyze?wait=true", files=files, data={"options": json.dumps(options or {})})
    assert r.status_code == 200, r.text
    return r.json()


async def test_health_and_stats(client: httpx.AsyncClient) -> None:
    health = (await client.get("/api/health")).json()
    assert health["status"] == "ok" and health["db"] is True and health["ready"] is True
    stats = (await client.get("/api/stats")).json()
    assert stats["media"] == 0 and stats["labels"] == []


async def test_plan_preview(client: httpx.AsyncClient) -> None:
    plan = (await client.post("/api/plan", json={"query": "dogs, menacing people", "expand": "false"})).json()
    assert plan["source"] == "literal" and plan["prompts"] == ["dog", "person"]
    assert plan["concepts"][1]["kind"] == "person_attribute"


async def test_analyze_wait_and_media_routes(client: httpx.AsyncClient) -> None:
    out = await _analyze(client, _scene(RED, BLUE), options={"query": "people, knives", "describe": "all"})
    job, (media,) = out["job"], out["media"]
    assert job["status"] == "done" and job["media_ids"] == [media["id"]] and job["plan"]["prompts"] == ["person", "knife"]
    assert media["kind"] == "image" and media["status"] == "done" and media["entity_count"] == 2 and media["person_count"] == 1
    assert media["plan"]["query"] == "people, knives"
    labels = {e["label"]: e for e in media["entities"]}
    assert labels["knife"]["description"] == "A knife in a blue object" and labels["knife"]["concept"] == "knives"
    assert labels["person"]["person"] is not None and labels["person"]["crop_url"]

    listing = (await client.get("/api/media")).json()
    assert listing["total"] == 1 and listing["media"][0]["id"] == media["id"]
    assert (await client.get("/api/media?kind=video")).json() == {"media": [], "total": 0}
    detail = (await client.get(f"/api/media/{media['id']}")).json()
    assert detail["entities"] == media["entities"]
    for url in (media["url"], media["thumb_url"], labels["person"]["crop_url"]):
        r = await client.get(url)
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    assert (await client.get("/api/media/m_missing")).status_code == 404

    job_detail = (await client.get(f"/api/jobs/{job['id']}")).json()
    assert [m["id"] for m in job_detail["media"]] == [media["id"]]
    assert (await client.get("/api/jobs")).json()["jobs"][0]["id"] == job["id"]
    events = await client.get(f"/api/jobs/{job['id']}/events")
    lines = [json.loads(line[6:]) for line in events.text.splitlines() if line.startswith("data: ")]
    assert lines[0]["type"] == "job_started" and lines[-1]["type"] == "job_done"
    assert (await client.post(f"/api/jobs/{job['id']}/cancel")).json()["job"]["status"] == "done"
    assert (await client.get("/api/jobs/j_missing")).status_code == 404


async def test_analyze_rejects_bad_input(client: httpx.AsyncClient) -> None:
    png = ("files", ("a.png", _scene(RED), "image/png"))
    assert (await client.post("/api/analyze", files=[png], data={"options": "{oops"})).status_code == 400
    assert (await client.post("/api/analyze", files=[png], data={"options": "[1]"})).status_code == 400
    r = await client.post("/api/analyze", files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400 and "unsupported" in r.json()["detail"]
    assert (await client.post("/api/analyze")).status_code == 422


async def test_entity_routes_and_questions(client: httpx.AsyncClient) -> None:
    media = (await _analyze(client, _scene(RED)))["media"][0]
    (ent,) = media["entities"]
    r = await client.post(f"/api/entities/{ent['id']}/ask", json={"question": "What is he wearing?"})
    assert r.status_code == 200 and r.json()["answer"] == "It looks like a test scene." and r.json()["scope"] == "crop"
    r = await client.post(f"/api/media/{media['id']}/ask", json={"question": "What happens here?"})
    assert r.status_code == 200
    detail = (await client.get(f"/api/entities/{ent['id']}")).json()
    assert detail["media"]["id"] == media["id"] and [q["question"] for q in detail["qa"]] == ["What is he wearing?"]
    assert [q["question"] for q in (await client.get(f"/api/media/{media['id']}/qa")).json()["qa"]] == ["What happens here?"]
    assert (await client.post(f"/api/entities/{ent['id']}/ask", json={"question": ""})).status_code == 422
    assert (await client.post("/api/entities/e_missing/ask", json={"question": "?"})).status_code == 404

    assert (await client.delete(f"/api/entities/{ent['id']}")).json() == {"success": True, "entity_id": ent["id"]}
    assert (await client.delete(f"/api/entities/{ent['id']}")).status_code == 404
    assert (await client.get("/api/persons")).json()["total"] == 0  # its unnamed cluster went with it


async def test_search_json_and_multipart(client: httpx.AsyncClient) -> None:
    await _analyze(client, _scene(RED, GREEN, BLUE))
    r = await client.post("/api/search", json={"text": "red jacket"})
    body = r.json()
    assert r.status_code == 200 and body["count"] >= 1 and body["query"] == {"text": "red jacket", "has_image": False}
    top = body["results"][0]
    assert top["entity"]["description"] == "A person in a red jacket" and top["media"]["kind"] == "image"
    knives = (await client.post("/api/search", json={"text": "weapons", "labels": ["knife"]})).json()
    assert [r["entity"]["label"] for r in knives["results"]] == ["knife"]
    single = (await client.post("/api/search", json={"text": "weapons", "labels": "knife"})).json()
    assert [r["entity"]["label"] for r in single["results"]] == ["knife"]
    assert (await client.post("/api/search", json={"text": "jacket", "kind": "video"})).json()["count"] == 0

    image = png_bytes(make_image([(GREEN, (100, 20, 160, 220))]))
    r = await client.post("/api/search", files={"file": ("q.png", image, "image/png")}, data={"limit": "5"})
    assert r.status_code == 200 and r.json()["query"]["has_image"] is True
    assert r.json()["results"][0]["entity"]["description"] == "A person in a green jacket"
    r = await client.post("/api/search", data={"text": "green"}, files={"file": ("q.png", image, "image/png")})
    assert r.json()["results"][0]["entity"]["description"] == "A person in a green jacket"

    for bad in (
        dict(json={}),
        dict(json=["red"]),
        dict(content=b"not json", headers={"content-type": "application/json"}),
        dict(json={"text": "red", "limit": "many"}),
        dict(data={"text": "red", "limit": "many"}, files={"file": ("q.png", image, "image/png")}),
        dict(files={"file": ("q.png", b"not an image", "image/png")}),
    ):
        assert (await client.post("/api/search", **bad)).status_code == 400, bad


async def test_persons_endpoints(client: httpx.AsyncClient) -> None:
    await _analyze(client, _scene(RED, GREEN))
    await _analyze(client, _scene(RED))
    persons = (await client.get("/api/persons")).json()
    assert persons["total"] == 2
    by_count = {p["sighting_count"]: p for p in persons["persons"]}
    red, green = by_count[2], by_count[1]
    assert red["media_count"] == 2 and red["thumb_url"].endswith("/crop")

    r = await client.patch(f"/api/persons/{red['id']}", json={"label": "Alice", "is_watchlist": True, "notes": "tall"})
    assert (r.json()["label"], r.json()["is_watchlist"], r.json()["notes"]) == ("Alice", True, "tall")
    assert (await client.patch(f"/api/persons/{red['id']}", json={"notes": ""})).json()["notes"] is None
    assert (await client.patch("/api/persons/p_missing", json={"label": "x"})).status_code == 404
    assert [p["id"] for p in (await client.get("/api/persons?watchlist=true")).json()["persons"]] == [red["id"]]
    assert (await client.get("/api/persons?q=alice")).json()["total"] == 1

    detail = (await client.get(f"/api/persons/{red['id']}")).json()
    assert len(detail["sightings"]) == 2 and all(s["media"]["kind"] == "image" for s in detail["sightings"])
    assert (await client.get("/api/persons/p_missing")).status_code == 404

    found = (await client.post("/api/persons/search", files={"file": ("q.png", _scene(RED), "image/png")})).json()
    assert found["results"][0]["person"]["id"] == red["id"] and found["results"][0]["similarity"] > 0.99

    merged = await client.post(f"/api/persons/{red['id']}/merge", json={"other_id": green["id"]})
    assert merged.status_code == 200 and merged.json()["sighting_count"] == 3 and merged.json()["label"] == "Alice"
    assert (await client.post(f"/api/persons/{red['id']}/merge", json={"other_id": red["id"]})).status_code == 400

    out = (await client.post("/api/persons/recluster", json={})).json()
    assert out["sightings"] == 3 and out["persons"] == 1  # merged sightings are manual anchors
    assert (await client.post("/api/persons/recluster")).status_code == 200

    assert (await client.delete(f"/api/persons/{red['id']}")).json()["success"] is True
    assert (await client.delete(f"/api/persons/{red['id']}")).status_code == 404


async def test_review_queue_and_assign(client: httpx.AsyncClient, db: Any) -> None:
    (await _analyze(client, _scene(RED)))
    (pid,) = [p["id"] for p in (await client.get("/api/persons")).json()["persons"]]
    mid = await add_media(db)
    pending, other = await add_entities(db, mid, [vec(1, 0, 1), vec(0, 1, 0)])
    await db.link_entity(pending, None, "pending", 0.7, suggested=pid)

    queue = (await client.get("/api/persons/review")).json()["pending"]
    assert [e["id"] for e in queue] == [pending] and queue[0]["suggested_person"]["id"] == pid
    r = await client.post(f"/api/entities/{pending}/review", json={"action": "accept"})
    assert r.status_code == 200 and r.json()["person"]["id"] == pid and r.json()["match_status"] == "manual"
    assert (await client.get("/api/persons/review")).json()["pending"] == []
    assert (await client.post(f"/api/entities/{pending}/review", json={"action": "accept"})).status_code == 400
    assert (await client.post(f"/api/entities/{pending}/review", json={"action": "bogus"})).status_code == 400
    assert (await client.post("/api/entities/e_missing/review", json={"action": "reject"})).status_code == 404

    r = await client.post(f"/api/entities/{other}/assign", json={"person_id": pid})
    assert r.json()["person"]["id"] == pid
    r = await client.post(f"/api/entities/{other}/assign", json={"person_id": None})
    assert r.json()["person"]["id"] != pid
    assert (await client.post(f"/api/entities/{other}/assign", json={"person_id": "p_missing"})).status_code == 404
    assert (await client.post("/api/entities/e_missing/assign", json={})).status_code == 404


async def test_delete_media_updates_persons(client: httpx.AsyncClient) -> None:
    first = (await _analyze(client, _scene(RED, GREEN)))["media"][0]
    await _analyze(client, _scene(RED))
    r = await client.delete(f"/api/media/{first['id']}")
    assert r.status_code == 200 and len(r.json()["persons_updated"]) == 2
    persons = (await client.get("/api/persons")).json()["persons"]
    assert [p["sighting_count"] for p in persons] == [1]  # green's only sighting was in the deleted image
    assert (await client.delete(f"/api/media/{first['id']}")).status_code == 404
    assert (await client.get("/api/stats")).json()["media"] == 1
