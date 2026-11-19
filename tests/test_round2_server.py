"""Round 2 server fixes: CLI analyze/reembed, start-up Re-ID migration, media status filter,
failed media ids on jobs, accepted-only analyze counts, person-only assignment, per-request
model spec validation, agent identity tools and RRF tie handling."""

from __future__ import annotations

import asyncio
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

import ovp.cli as cli
import ovp.context as context_mod
from ovp.api.app import app
from ovp.jobs import InputItem, JobManager
from ovp.models.llm import ChatResponse, ToolCall
from ovp.pipeline.analyzer import AnalyzeOptions
from ovp.pipeline.planner import QueryPlanner
from ovp.services.search import SearchService, dense_ranks
from ovp.storage.db import Database

from .fakes import BLUE, RED, FakeVLM, make_image, png_bytes
from .seed import add_entities, add_media
from .test_agent import ScriptedLLM, call, chat, say


@pytest.fixture
async def client(app_ctx: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _scene(*colors: tuple[int, int, int]) -> bytes:
    return png_bytes(make_image([(c, (20 + 90 * i, 30, 90 + 90 * i, 200)) for i, c in enumerate(colors)]))


@pytest.fixture
def cli_ctx(app_ctx: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """CLI commands run against the test context instead of creating their own."""

    @asynccontextmanager
    async def fake_context() -> AsyncIterator[Any]:
        yield app_ctx

    monkeypatch.setattr(cli, "app_context", fake_context)
    return app_ctx


@pytest.fixture
def reembed_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []

    async def fake_reembed(db: Any, store: Any, reid: Any) -> dict:
        calls.append((db, store, reid))
        return {"sightings": 3, "persons": 1}

    monkeypatch.setattr(context_mod, "reembed_orphaned", fake_reembed)
    return calls


# ── CLI ───────────────────────────────────────────────────────────────────────


async def test_cli_analyze_folder_outside_import_roots(cli_ctx: Any, tmp_path: Path) -> None:
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "a.png").write_bytes(_scene(RED, BLUE))
    (folder / "notes.txt").write_text("not media")
    args = cli.build_parser().parse_args(["analyze", str(folder), "--json", "--quiet"])
    out = io.StringIO()
    assert await cli.run_analyze(args, out) == 0  # OVP_IMPORT_ROOTS is unset: the CLI is not restricted
    res = json.loads(out.getvalue())
    assert res["job"]["status"] == "done" and [m["name"] for m in res["media"]] == ["a.png"]
    assert sorted(e["label"] for e in res["media"][0]["entities"]) == ["knife", "person"]


async def test_cli_reembed(cli_ctx: Any, reembed_calls: list[tuple]) -> None:
    out = io.StringIO()
    assert await cli.run_reembed(cli.build_parser().parse_args(["reembed", "--json"]), out) == 0
    assert json.loads(out.getvalue()) == {"space": "reid:osnet:fake", "sightings": 3, "persons": 1}
    assert reembed_calls == [(cli_ctx.db, cli_ctx.store, cli_ctx.registry.models["reid"])]

    cli_ctx.registry.models["reid"] = None
    assert await cli.run_reembed(cli.build_parser().parse_args(["reembed"]), io.StringIO()) == 2


async def test_startup_runs_reembed_after_warmup(app_ctx: Any, reembed_calls: list[tuple]) -> None:
    app_ctx.ready = False
    await app_ctx.startup(load_models=True)
    assert app_ctx.ready and len(reembed_calls) == 1

    app_ctx.registry.models["reid"] = None  # no Re-ID model: nothing to migrate
    await app_ctx.startup(load_models=False)
    assert len(reembed_calls) == 1


# ── media listing & jobs ──────────────────────────────────────────────────────


async def test_media_status_filter_and_matching_total(client: httpx.AsyncClient, db: Database) -> None:
    ids = {}
    for status in ("done", "processing", "error", "cancelled"):
        ids[status] = await add_media(db)
        await db.update_media(ids[status], status=status)
    listing = (await client.get("/api/media")).json()
    assert {m["id"] for m in listing["media"]} == {ids["done"], ids["processing"]} and listing["total"] == 2
    listing = (await client.get("/api/media", params={"status": "error, cancelled"})).json()
    assert {m["id"] for m in listing["media"]} == {ids["error"], ids["cancelled"]} and listing["total"] == 2
    assert (await client.get("/api/media", params={"status": "error", "kind": "video"})).json() == {"media": [], "total": 0}
    for bad in ("pending", "done,bogus", ","):
        assert (await client.get("/api/media", params={"status": bad})).status_code == 400


async def test_cancelled_media_is_recorded_on_the_job(app_ctx: Any, client: httpx.AsyncClient, tmp_path: Path) -> None:
    jobs: JobManager = app_ctx.jobs
    detector = app_ctx.registry.models["detector"]
    detector.gate = asyncio.Event()
    path = jobs.tmp_path("a.png")
    path.write_bytes(_scene(RED))
    job = jobs.submit_analyze([InputItem(path, "a.png", "image")], AnalyzeOptions())
    for _ in range(500):
        if any(e["type"] == "media_created" for e in job.events):
            break
        await asyncio.sleep(0.01)
    jobs.cancel(job)
    detector.gate.set()
    await jobs.wait(job, timeout=30)
    created = next(e["media"]["id"] for e in job.events if e["type"] == "media_created")
    assert job.status == "cancelled" and job.media_ids == [] and job.to_dict()["failed_media_ids"] == [created]
    listing = (await client.get("/api/media", params={"status": "cancelled"})).json()
    assert [m["id"] for m in listing["media"]] == [created]


async def test_analyze_summary_counts_only_accepted_entities(app_ctx: Any, tmp_path: Path) -> None:
    app_ctx.registry.models["vlm"] = FakeVLM(answer="no")
    path = tmp_path / "k.png"
    path.write_bytes(_scene(BLUE, RED))
    job = app_ctx.jobs.submit_analyze([InputItem(path, "k.png", "image")], AnalyzeOptions(query="weapons", describe="none"))
    await app_ctx.jobs.wait(job)
    (media,) = await app_ctx.jobs.media_details(job.media_ids)
    assert [e["stage"] for e in media["entities"]] == ["rejected"]
    assert job.summary == {"media": 1, "entities": 0, "failed": 0} and media["entity_count"] == 0


def test_spool_paths_use_whitelisted_suffixes(app_ctx: Any) -> None:
    jobs: JobManager = app_ctx.jobs
    assert jobs.tmp_path("weights.pt", "video").suffix == ".mp4"
    assert jobs.tmp_path("clip.MOV", "video").suffix == ".mov"
    assert jobs.tmp_path("weights.pt", "image").suffix == ".jpg"
    assert jobs.tmp_path("photo.PNG").suffix == ".png"
    assert jobs.tmp_path("", "video").suffix == ".mp4"


# ── entities & options ────────────────────────────────────────────────────────


async def test_assign_rejects_non_person_entities(client: httpx.AsyncClient, db: Database) -> None:
    mid = await add_media(db)
    (tie,) = await add_entities(db, mid, [None], label="tie")
    (person,) = await add_entities(db, mid, [None])
    r = await client.post(f"/api/entities/{tie}/assign", json={"person_id": None})
    assert r.status_code == 400 and "person" in r.json()["detail"]
    r = await client.post(f"/api/entities/{person}/assign", json={"person_id": None})
    assert r.status_code == 200 and r.json()["person"] is not None


async def test_invalid_model_specs_are_rejected_before_any_media(client: httpx.AsyncClient, app_ctx: Any, db: Database, tmp_path: Path) -> None:
    bad = {"detector": "yolo:/etc/x.pt"}
    r = await client.post("/api/analyze", files=[("files", ("a.png", _scene(RED), "image/png"))], data={"options": json.dumps(bad)})
    assert r.status_code == 400 and "working directory" in r.json()["detail"]
    r = await client.post("/api/analyze/paths", json={"paths": [str(tmp_path)], "options": {"vlm": "bogus:model"}})
    assert r.status_code == 400 and "bogus" in r.json()["detail"]
    r = await client.post("/api/scan", json={"query": "people", "options": {"segmenter": "sam:/etc/sam.pt"}})
    assert r.status_code == 400
    r = await client.post("/api/scan", json={"query": "people", "options": {"llm": 5}})
    assert r.status_code == 400
    assert app_ctx.jobs.jobs == {}
    assert await db.count_media(statuses=("done", "processing", "error", "cancelled")) == 0
    assert list(app_ctx.jobs.tmp_dir.iterdir()) == []


async def test_agent_chat_accepts_the_configured_default_llm(client: httpx.AsyncClient, app_ctx: Any) -> None:
    app_ctx.registry.models["llm"] = ScriptedLLM(say("hello"))
    app_ctx.registry.defaults["llm"] = "qwen:/opt/models/qwen-vl"  # operator-configured, outside the weight roots
    r = await client.post("/api/agent/chat", json={"message": "hi", "llm": "qwen:/opt/models/qwen-vl"})
    assert r.status_code == 200 and '"message"' in r.text
    r = await client.post("/api/agent/chat", json={"message": "hi", "llm": "qwen:/opt/models/other"})
    assert r.status_code == 400


# ── agent identity tools ──────────────────────────────────────────────────────


async def test_agent_merge_goes_through_identity_service(client: httpx.AsyncClient, app_ctx: Any, db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    mid = await add_media(db)
    a, b = await add_entities(db, mid, [None, None])
    pa, pb = await db.create_person("manual", None, a), await db.create_person("manual", None, b)
    merged: list[tuple[str, str]] = []
    real_merge = app_ctx.identity.merge

    async def spy(keep: str, other: str) -> bool:
        merged.append((keep, other))
        assert not app_ctx.identity.lock.locked()
        return await real_merge(keep, other)

    monkeypatch.setattr(app_ctx.identity, "merge", spy)
    app_ctx.registry.models["llm"] = ScriptedLLM(call("merge_persons", keep_id=pa, other_id=pb), say("merged"))
    await chat(client, "merge them")
    assert merged == [(pa, pb)] and await db.get_person(pb) is None


async def test_agent_find_person_sightings_prefers_exact_name(client: httpx.AsyncClient, app_ctx: Any, db: Database) -> None:
    mid = await add_media(db)
    a, b = await add_entities(db, mid, [None, None])
    ann, anna = await db.create_person("manual", None, a), await db.create_person("manual", None, b)
    await db.update_person(ann, label="Ann")
    await db.update_person(anna, label="Anna")
    llm = ScriptedLLM(ChatResponse(text="", tool_calls=[ToolCall("call_1", "find_person_sightings", {"name": " ann "})]), say("found"))
    app_ctx.registry.models["llm"] = llm
    await chat(client, "where is ann?")
    tool = json.loads(next(m for m in llm.requests[1] if m["role"] == "tool")["content"])
    assert tool["person"]["id"] == ann


# ── search fusion ─────────────────────────────────────────────────────────────


def test_dense_ranks() -> None:
    assert dense_ranks([]) == []
    assert dense_ranks([1.0, 1.0, 1.0]) == [0, 0, 0]
    assert dense_ranks([1.2, 0.8, 0.8, 0.5]) == [0, 1, 1, 2]


class _SignalDB:
    """Database stand-in returning fixed signal lists."""

    def __init__(self, fts: list, label: list):
        self._fts, self._label = fts, label

    async def fts(self, *_: Any) -> list:
        return self._fts

    async def label_match(self, *_: Any) -> list:
        return self._label


async def test_tied_label_scores_do_not_reorder_results(registry: Any) -> None:
    registry.models["text_embedder"] = None
    label = [("e_a", 1.0), ("e_b", 1.0), ("e_c", 1.0)]  # every person matches "man"
    fts = [("e_c", 1.2), ("e_b", 0.8), ("e_a", 0.8)]
    svc = SearchService(_SignalDB(fts, label), registry, QueryPlanner())  # type: ignore[arg-type]
    hits = await svc.search("man in a suit", limit=3)
    assert hits[0]["entity_id"] == "e_c"  # best fts match first; position in the label list is irrelevant
    assert hits[1]["score"] == hits[2]["score"]  # tied on both signals
