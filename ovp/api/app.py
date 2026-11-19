"""FastAPI application. Endpoint contract: docs/API.md."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from .. import __version__, serialize
from ..config import get_settings
from ..context import AppContext, create_context, get_context
from ..jobs import InputItem
from ..models.registry import ModelUnavailable
from ..pipeline.analyzer import AnalyzeOptions
from ..storage.files import media_kind

log = logging.getLogger("ovp.api")


# ──────────────────────────────────────────────────────────────────────────────
# lifespan
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ctx = await create_context()
    if get_settings().load_models_on_startup:
        # load models in the background: the API (library, search, history) is usable immediately
        ctx.warmup_task = asyncio.create_task(ctx.warmup())
    else:
        ctx.ready = True
    log.info("OpenVocab Vision API up · device=%s · defaults=%s", ctx.registry.device, ctx.registry.defaults)
    yield
    for job in ctx.jobs.jobs.values():
        ctx.jobs.cancel(job)
    if ctx.warmup_task:
        ctx.warmup_task.cancel()
    await ctx.registry.aclose()
    await ctx.db.close()


app = FastAPI(title="OpenVocab Vision API", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2048)


def ctx() -> AppContext:
    return get_context()


@app.exception_handler(KeyError)
async def _key_error(_: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse({"detail": f"not found: {exc.args[0] if exc.args else ''}"}, status_code=404)


def sse(gen: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    async def body() -> AsyncIterator[bytes]:
        async for ev in gen:
            yield f"data: {json.dumps(ev, default=str)}\n\n".encode()

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


async def read_upload_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# system
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health() -> dict[str, Any]:
    c = ctx()
    return {
        "status": "ok",
        "ready": c.ready,
        "version": __version__,
        "device": c.registry.device,
        "db": await c.db.ping(),
        "capabilities": c.registry.capabilities(),
    }


@app.get("/health")  # backwards compatible
async def health_legacy() -> dict[str, Any]:
    h = await health()
    return {**h, "pipeline_ready": h["ready"], "reid_ready": bool(h["capabilities"]["reid"])}


@app.get("/api/models")
async def models() -> dict[str, Any]:
    return ctx().registry.report()


class ConfigPatch(BaseModel):
    detector: Optional[str] = None
    segmenter: Optional[str] = None
    vlm: Optional[str] = None
    llm: Optional[str] = None
    embedder: Optional[str] = None
    text_embedder: Optional[str] = None
    reid: Optional[str] = None


@app.patch("/api/config")
async def patch_config(body: ConfigPatch) -> dict[str, Any]:
    c = ctx()
    try:
        await c.registry.switch_defaults(body.model_dump(exclude_none=True))
    except ModelUnavailable as e:
        raise HTTPException(400, str(e)) from e
    return c.registry.report()


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    return await ctx().db.stats()


# ──────────────────────────────────────────────────────────────────────────────
# planning / analysis / jobs
# ──────────────────────────────────────────────────────────────────────────────


class PlanRequest(BaseModel):
    query: Optional[str] = None
    expand: Any = "auto"
    llm: Optional[str] = None


@app.post("/api/plan")
async def plan(body: PlanRequest) -> dict[str, Any]:
    p = await ctx().analyzer.make_plan(AnalyzeOptions.from_dict(body.model_dump()))
    return p.to_dict()


def parse_options(raw: Optional[str | dict]) -> AnalyzeOptions:
    if raw is None or raw == "":
        return AnalyzeOptions()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"options must be JSON: {e}")
    if not isinstance(raw, dict):
        raise HTTPException(400, "options must be a JSON object")
    try:
        return AnalyzeOptions.from_dict(raw)
    except TypeError as e:
        raise HTTPException(400, str(e))


async def job_payload(job: Any, with_media: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"job": job.to_dict()}
    if with_media:
        c = ctx()
        media = []
        for mid in job.media_ids:
            m = await c.db.get_media(mid)
            if m is not None:
                media.append(serialize.media_detail(m, await c.db.media_entities(mid)))
        out["media"] = media
    return out


@app.post("/api/analyze")
async def analyze(
    files: list[UploadFile] = File(...),
    options: Optional[str] = Form(None),
    wait: bool = Query(False),
    query: Optional[str] = Form(None),
) -> dict[str, Any]:
    c = ctx()
    opts = parse_options(options)
    if query and not opts.query:
        opts.query = query
    items: list[InputItem] = []
    try:
        for f in files:
            kind = media_kind(f.filename or "", f.content_type)
            if kind is None:
                raise HTTPException(400, f"unsupported file type: {f.filename}")
            path = c.jobs.tmp_path(f.filename or ("upload.mp4" if kind == "video" else "upload.jpg"))
            with open(path, "wb") as out:
                while chunk := await f.read(1 << 20):
                    out.write(chunk)
            items.append(InputItem(path, f.filename or path.name, kind))
    except BaseException:
        for it in items:
            it.path.unlink(missing_ok=True)
        raise
    if not items:
        raise HTTPException(400, "no files")
    job = c.jobs.submit_analyze(items, opts)
    if wait:
        await c.jobs.wait(job)
        return await job_payload(job)
    return {"job": job.to_dict()}


class PathsRequest(BaseModel):
    paths: list[str]
    options: Optional[dict[str, Any]] = None


@app.post("/api/analyze/paths")
async def analyze_paths(body: PathsRequest, wait: bool = Query(False)) -> dict[str, Any]:
    c = ctx()
    items = c.jobs.collect_paths(body.paths)
    if not items:
        raise HTTPException(400, "no images or videos found at the given paths")
    job = c.jobs.submit_analyze(items, parse_options(body.options))
    if wait:
        await c.jobs.wait(job)
        return await job_payload(job)
    return {"job": job.to_dict()}


class ScanRequest(BaseModel):
    query: str
    media_ids: Optional[list[str]] = None
    limit: Optional[int] = None
    options: Optional[dict[str, Any]] = None


@app.post("/api/scan")
async def scan(body: ScanRequest, wait: bool = Query(False)) -> dict[str, Any]:
    c = ctx()
    opts = parse_options({"describe": "none", **(body.options or {}), "query": body.query})
    job = c.jobs.submit_scan(body.media_ids, opts, body.limit)
    if wait:
        await c.jobs.wait(job)
        return await job_payload(job)
    return {"job": job.to_dict()}


@app.get("/api/jobs")
async def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.to_dict() for j in ctx().jobs.list()]}


def _job(job_id: str) -> Any:
    job = ctx().jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    return await job_payload(_job(job_id))


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    job = _job(job_id)
    return sse(ctx().jobs.subscribe(job))


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, Any]:
    job = _job(job_id)
    ctx().jobs.cancel(job)
    return {"job": job.to_dict()}


# ──────────────────────────────────────────────────────────────────────────────
# media & entities
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/api/media")
async def list_media(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), kind: Optional[str] = None) -> dict[str, Any]:
    c = ctx()
    rows = await c.db.list_media(limit, offset, kind)
    return {"media": [serialize.media(r) for r in rows], "total": await c.db.count_media(kind)}


async def _media(media_id: str) -> Any:
    m = await ctx().db.get_media(media_id)
    if m is None:
        raise HTTPException(404, "media not found")
    return m


@app.get("/api/media/{media_id}")
async def get_media(media_id: str) -> dict[str, Any]:
    m = await _media(media_id)
    return serialize.media_detail(m, await ctx().db.media_entities(media_id))


@app.delete("/api/media/{media_id}")
async def delete_media(media_id: str) -> dict[str, Any]:
    c = ctx()
    res = await c.db.delete_media(media_id)
    if res is None:
        raise HTTPException(404, "media not found")
    await c.store.delete(res["paths"])
    return {"success": True, "media_id": media_id, "persons_updated": res["persons"]}


VIDEO_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo", ".m4v": "video/mp4"}


@app.get("/api/media/{media_id}/file")
async def media_file(media_id: str) -> FileResponse:
    m = await _media(media_id)
    p = Path(m["path"])
    if not p.exists():
        raise HTTPException(404, "file missing")
    mt = "image/jpeg" if m["kind"] == "image" and p.suffix.lower() in (".jpg", ".jpeg") else VIDEO_TYPES.get(p.suffix.lower())
    return FileResponse(p, media_type=mt, headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/media/{media_id}/thumb")
async def media_thumb(media_id: str) -> FileResponse:
    m = await _media(media_id)
    p = Path(m["thumb_path"] or m["path"])
    if not p.exists():
        raise HTTPException(404, "thumbnail missing")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    scope: str = "crop"
    use_full_scene: bool = False
    vlm: Optional[str] = None


@app.post("/api/media/{media_id}/ask")
async def ask_media(media_id: str, body: AskRequest) -> dict[str, Any]:
    t = time.perf_counter()
    try:
        ans = await ctx().analyzer.ask_media(media_id, body.question, body.vlm)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"answer": ans, "elapsed": round(time.perf_counter() - t, 2)}


@app.get("/api/media/{media_id}/qa")
async def media_qa(media_id: str) -> dict[str, Any]:
    rows = await ctx().db.list_qa(media_id=media_id)
    return {"qa": [{"question": r["question"], "answer": r["answer"], "created_at": r["created_at"].timestamp()} for r in rows]}


async def _entity(entity_id: str) -> Any:
    e = await ctx().db.get_entity(entity_id)
    if e is None:
        raise HTTPException(404, "entity not found")
    return e


@app.get("/api/entities/{entity_id}")
async def get_entity(entity_id: str) -> dict[str, Any]:
    c = ctx()
    e = await _entity(entity_id)
    m = await c.db.get_media(e["media_id"])
    qa = await c.db.list_qa(entity_id=entity_id)
    return {
        **serialize.entity(e),
        "media": serialize.media(m) if m else None,
        "qa": [{"question": r["question"], "answer": r["answer"], "scope": r["scope"], "created_at": r["created_at"].timestamp()} for r in qa],
    }


@app.delete("/api/entities/{entity_id}")
async def delete_entity(entity_id: str) -> dict[str, Any]:
    c = ctx()
    crop = await c.db.delete_entity(entity_id)
    if crop is None:
        raise HTTPException(404, "entity not found")
    await c.store.delete([crop])
    return {"success": True, "entity_id": entity_id}


@app.get("/api/entities/{entity_id}/crop")
async def entity_crop(entity_id: str) -> FileResponse:
    e = await _entity(entity_id)
    p = Path(e["crop_path"] or "")
    if not e["crop_path"] or not p.exists():
        raise HTTPException(404, "crop missing")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})


@app.post("/api/entities/{entity_id}/ask")
async def ask_entity(entity_id: str, body: AskRequest) -> dict[str, Any]:
    scope = "scene" if body.use_full_scene else body.scope
    t = time.perf_counter()
    try:
        ans = await ctx().analyzer.ask_entity(entity_id, body.question, scope, body.vlm)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    return {"answer": ans, "elapsed": round(time.perf_counter() - t, 2), "scope": scope}


class AssignRequest(BaseModel):
    person_id: Optional[str] = None


@app.post("/api/entities/{entity_id}/assign")
async def assign_entity(entity_id: str, body: AssignRequest) -> dict[str, Any]:
    c = ctx()
    await _entity(entity_id)
    if body.person_id and await c.db.get_person(body.person_id) is None:
        raise HTTPException(404, "person not found")
    await c.identity.assign(entity_id, body.person_id)
    return serialize.entity(await _entity(entity_id))


class ReviewRequest(BaseModel):
    action: str


@app.post("/api/entities/{entity_id}/review")
async def review_entity(entity_id: str, body: ReviewRequest) -> dict[str, Any]:
    try:
        await ctx().identity.review(entity_id, body.action)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return serialize.entity(await _entity(entity_id))


# ──────────────────────────────────────────────────────────────────────────────
# search
# ──────────────────────────────────────────────────────────────────────────────


def _search_limit(raw: Any) -> int:
    try:
        return max(1, min(int(raw or 30), 200))
    except (TypeError, ValueError):
        raise HTTPException(400, "limit must be an integer")


@app.post("/api/search")
async def search(request: Request) -> dict[str, Any]:
    c = ctx()
    ctype = request.headers.get("content-type", "")
    image = None
    filters: dict[str, Any] = {}
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        text = form.get("text") or form.get("text_query") or None
        limit = _search_limit(form.get("limit"))
        f = form.get("file")
        if f is not None and hasattr(f, "read"):
            image = await read_upload_image(f)  # type: ignore[arg-type]
        for k in ("media_id", "person_id", "kind"):
            if form.get(k):
                filters[k] = form.get(k)
    else:
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            raise HTTPException(400, "send a JSON object {text, ...} or multipart with text/file")
        text = body.get("text") or body.get("text_query")
        limit = _search_limit(body.get("limit"))
        filters = {k: body.get(k) for k in ("labels", "media_id", "person_id", "kind") if body.get(k)}
        if isinstance(filters.get("labels"), str):
            filters["labels"] = [filters["labels"]]
    if not (text and str(text).strip()) and image is None:
        raise HTTPException(400, "provide text and/or an image")
    hits = await c.search.search(str(text) if text else None, image, filters, limit)
    results = await c.search.hydrate(hits)
    return {"results": results, "count": len(results), "query": {"text": text, "has_image": image is not None}}


# ──────────────────────────────────────────────────────────────────────────────
# persons
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/api/persons")
async def list_persons(watchlist: bool = False, q: Optional[str] = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0), order: str = "recent") -> dict[str, Any]:
    rows, total = await ctx().db.list_persons(watchlist, q, limit, offset, order)
    return {"persons": [serialize.person(r) for r in rows], "total": total}


@app.get("/api/persons/review")
async def review_queue(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    rows = await ctx().db.pending_entities(limit)
    return {"pending": [serialize.entity(r) for r in rows]}


class ReclusterRequest(BaseModel):
    threshold: Optional[float] = None


@app.post("/api/persons/recluster")
async def recluster(body: Optional[ReclusterRequest] = None) -> dict[str, Any]:
    c = ctx()
    space = c.reid_space()
    if not space:
        raise HTTPException(503, "no Re-ID model available")
    return await c.identity.recluster(space, body.threshold if body else None)


@app.post("/api/persons/search")
async def persons_by_photo(file: UploadFile = File(...), limit: int = Query(15, ge=1, le=100), min_similarity: float = Query(0.2)) -> dict[str, Any]:
    c = ctx()
    reid = await c.registry.ensure("reid")
    if reid is None:
        raise HTTPException(503, "no Re-ID model available")
    img = await read_upload_image(file)
    crop = await c.search.person_crop(img)
    vec = (await reid.embed([crop]))[0]
    matches = [(pid, s) for pid, s in await c.identity.search_by_vector(reid.space, vec, limit) if s >= min_similarity]
    rows = {r["id"]: r for r in await c.db.get_persons([p for p, _ in matches])}
    return {"results": [{"person": serialize.person(rows[p]), "similarity": round(s, 4)} for p, s in matches if p in rows]}


async def _person(person_id: str) -> Any:
    p = await ctx().db.get_person(person_id)
    if p is None:
        raise HTTPException(404, "person not found")
    return p


@app.get("/api/persons/{person_id}")
async def get_person(person_id: str) -> dict[str, Any]:
    c = ctx()
    p = await _person(person_id)
    sights = await c.db.person_sightings(person_id)
    medias = {m["id"]: serialize.media(m) for m in await c.db.get_media_many(list({s["media_id"] for s in sights}))}
    return {**serialize.person(p), "sightings": [{**serialize.entity(s), "media": medias.get(s["media_id"])} for s in sights]}


class PersonPatch(BaseModel):
    label: Optional[str] = None
    is_watchlist: Optional[bool] = None
    notes: Optional[str] = None


@app.patch("/api/persons/{person_id}")
async def patch_person(person_id: str, body: PersonPatch) -> dict[str, Any]:
    c = ctx()
    await _person(person_id)
    fields = body.model_dump(exclude_unset=True)
    # empty strings clear label/notes
    sets = {k: (v if v != "" else None) for k, v in fields.items()}
    if sets:
        assert c.db.pool
        cols = ", ".join(f"{k} = ${i + 2}" for i, k in enumerate(sets))
        await c.db.pool.execute(f"UPDATE persons SET {cols}, updated_at = NOW() WHERE id = $1", person_id, *sets.values())
    return serialize.person(await _person(person_id))


@app.delete("/api/persons/{person_id}")
async def delete_person(person_id: str) -> dict[str, Any]:
    if not await ctx().db.delete_person(person_id):
        raise HTTPException(404, "person not found")
    return {"success": True, "person_id": person_id}


class MergeRequest(BaseModel):
    other_id: str


@app.post("/api/persons/{person_id}/merge")
async def merge_persons(person_id: str, body: MergeRequest) -> dict[str, Any]:
    c = ctx()
    if not await c.db.merge_persons(person_id, body.other_id):
        raise HTTPException(400, "merge failed (check both ids)")
    return serialize.person(await _person(person_id))


# ──────────────────────────────────────────────────────────────────────────────
# agent
# ──────────────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: Optional[str] = None
    llm: Optional[str] = None


@app.post("/api/agent/chat")
async def agent_chat(body: ChatRequest) -> StreamingResponse:
    c = ctx()
    llm = await c.registry.ensure("llm", body.llm)
    if llm is None:
        if body.llm:
            reason = c.registry.errors.get(f"llm|{body.llm}", "not configured")
            raise HTTPException(400, f"LLM {body.llm!r} is unavailable: {reason}")
        raise HTTPException(503, "no LLM configured for the agent – set OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY or OVP_COMPAT_BASE_URL")
    if not llm.supports_tools:
        raise HTTPException(400, f"{llm.spec} does not support tool calling")
    conv = c.agent.get_conversation(body.conversation_id)
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def emit(type: str, **data: Any) -> None:
        await queue.put({"type": type, **data})

    def finished(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.error("agent turn failed", exc_info=task.exception())
            queue.put_nowait({"type": "error", "message": str(task.exception())})
        queue.put_nowait({"type": "done"})

    async def stream() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "conversation", "conversation_id": conv.id}
        # The turn runs independently of this response: a client disconnect stops the stream,
        # not the turn, so the conversation stays consistent.
        c.agent.start_turn(conv, body.message, llm, emit).add_done_callback(finished)
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield {"type": "ping"}
                continue
            yield ev
            if ev["type"] == "done":
                break

    return sse(stream())


@app.get("/api/agent/conversations")
async def list_conversations() -> dict[str, Any]:
    return {"conversations": ctx().agent.list_conversations()}


@app.get("/api/agent/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    c = ctx()
    conv = c.agent.conversations.get(conversation_id)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    return {"id": conv.id, "messages": c.agent.transcript(conv)}


# ──────────────────────────────────────────────────────────────────────────────
# frontend (React build)
# ──────────────────────────────────────────────────────────────────────────────

_dist = get_settings().frontend_dist
if (_dist / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
async def spa(path: str) -> Any:
    if path.startswith("api/"):
        raise HTTPException(404, "not found")
    dist = get_settings().frontend_dist
    target = (dist / path).resolve()
    if path and target.is_file() and dist.resolve() in target.parents:
        return FileResponse(target)
    index = dist / "index.html"
    if index.exists():
        return FileResponse(index, headers={"Cache-Control": "no-cache"})
    return HTMLResponse(
        "<h1>OpenVocab Vision API</h1><p>The React UI is not built. Run <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code> "
        "or use the dev server (<code>npm run dev</code>, http://localhost:5173). API docs: <a href='/docs'>/docs</a></p>"
    )
