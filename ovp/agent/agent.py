"""Analyst agent: an LLM with tools over the media library.

The agent can search, run new open-vocabulary scans over stored photos/videos, inspect and
question images with the VLM, cluster / name / merge people, and ingest folders. Tool results
carry full entity/media/person objects for the UI (rendered as galleries) and a compact JSON
summary for the model.

Conversations live in memory. A turn runs as its own task so it finishes (and the transcript
stays valid) even when the client disconnects mid-stream; turns of one conversation are
serialized by a lock, different conversations run concurrently.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .. import serialize
from ..models.llm import ChatModel, ToolCall, ToolSpec
from ..storage.db import new_id

log = logging.getLogger("ovp.agent")

Emit = Callable[..., Awaitable[None]]
ToolFn = Callable[..., Awaitable["ToolOutput"]]

MAX_CONVERSATIONS = 200
MAX_TOOL_RESULT_CHARS = 14_000
HISTORY_BUDGET_CHARS = 120_000

SYSTEM_PROMPT = """You are OpenVocab Analyst, an assistant that investigates a private library of photos and videos.
Every photo/video has been processed: objects were detected (open-vocabulary), people were described by a vision-language model (VLM) and grouped into identity clusters ("persons") via Re-ID.

Ground rules:
- Answer only from tool results; never invent entities, people, counts, ids or file names.
- Whenever the user mentions an id (m_..., e_..., p_..., j_...), call a tool with it — even if it looks malformed. Never say an id does not exist, or that the library lacks something, without a tool result showing it.
- Resolve references like "that person" / "him" / "those" from the conversation (the item most recently discussed). Ask which one only if it is genuinely ambiguous — e.g. several people tie, or a name matches several persons — especially before renaming, watchlisting or merging.

Choosing tools:
- Things already detected or described (objects, people's clothing/actions/appearance): `search` first.
- Counting objects across the library: `stats` (label counts); `search` results are capped and are not a count.
- Something that may not have been detected yet (e.g. "dangerous things", "all living things", a specific object): `find_in_library`, which re-scans stored media with an open-vocabulary detector (LLM-expanded vocabulary + VLM verification for abstract requests). It is slower; tell the user what was checked.
- "What is happening in m_..." or any question about the content of one photo/video or detection: `ask_image`.
- People: `list_persons` (order "sightings" for who appears most), `get_person`, `find_person_sightings`, `name_person`, `set_watchlist`, `merge_persons` (only when the user asks or the evidence is strong — say why), `recluster_persons`.

Answers: refer to people by name, or as "an unnamed person" (with a distinguishing detail) when they have no name, and to photos/videos by file name. Do not show raw ids (e_..., m_..., p_..., j_...) unless the user asks for them: the UI already shows clickable galleries of the returned items, so do not repeat every field either. Keep answers concise with short bullet lists, and state uncertainty honestly (Re-ID matches and VLM judgements are probabilistic)."""

SUMMARIZE_PROMPT = "Tool budget exhausted. Without calling more tools, summarize for the user what you found so far."


@dataclass
class Conversation:
    id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def title(self) -> str:
        first = next((m["content"] for m in self.messages if m["role"] == "user"), "")
        return first[:80]


@dataclass
class ToolOutput:
    summary: str
    data: Any = None  # sent to the model (compact)
    entities: list[dict] = field(default_factory=list)
    media: list[dict] = field(default_factory=list)
    persons: list[dict] = field(default_factory=list)
    job_id: Optional[str] = None
    is_error: bool = False  # the tool failed (providers that support it are told so)


def _fail(message: str) -> ToolOutput:
    return ToolOutput(message, {"error": message}, is_error=True)


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


class Agent:
    def __init__(self, ctx: Any):
        """``ctx`` is the app context (db, analyzer, search, identity, jobs, registry, store)."""
        self.ctx = ctx
        self.conversations: dict[str, Conversation] = {}
        self.tools: dict[str, tuple[ToolSpec, ToolFn]] = {}
        self._turns: set[asyncio.Task] = set()  # strong refs: turns outlive their HTTP request
        self._register()

    # ── tool registry ─────────────────────────────────────────────────────────
    def _tool(self, name: str, description: str, params: dict[str, Any], required: list[str] | None = None) -> Callable[[ToolFn], ToolFn]:
        def deco(fn: ToolFn) -> ToolFn:
            self.tools[name] = (ToolSpec(name, description, _obj(params, required)), fn)
            return fn

        return deco

    def _register(self) -> None:
        t = self._tool
        S = {"type": "string"}
        I = {"type": "integer"}
        db = self.ctx.db

        async def person_out(person_id: str, summary: str) -> ToolOutput:
            p = await db.get_person(person_id)
            if p is None:
                return _fail(f"person {person_id} not found")
            pd = serialize.person(p)
            return ToolOutput(summary, {"person": _compact_person(pd)}, persons=[pd])

        @t("search", "Hybrid search over already-detected entities (labels, VLM descriptions of people's clothing/actions/appearance, CLIP). Returns the best matching detections with their media; results are capped, so do not use them as a count.",
           {"query": {**S, "description": "what to look for, e.g. 'man in red jacket', 'knife', 'people on bikes'"},
            "labels": {"type": "array", "items": S, "description": "optional exact detector labels to restrict to, e.g. ['person']"},
            "media_id": {**S, "description": "restrict to one photo/video"}, "person_id": {**S, "description": "restrict to one person"},
            "limit": {**I, "description": "max results (default 12, max 40)"}}, ["query"])
        async def search(query: str, labels: Optional[list[str]] = None, media_id: Optional[str] = None, person_id: Optional[str] = None, limit: int = 12) -> ToolOutput:
            hits = await self.ctx.search.search(query, None, {"labels": labels, "media_id": media_id, "person_id": person_id}, min(max(limit, 1), 40))
            res = await self.ctx.search.hydrate(hits)
            ents = [r["entity"] for r in res]
            media = _uniq([r["media"] for r in res if r["media"]])
            found = [{**serialize.compact_entity(r["entity"]), "media_name": (r["media"] or {}).get("name"), "score": round(r["score"], 3)} for r in res]
            return ToolOutput(f"{len(ents)} result(s)", {"results": found}, ents, media)

        @t("find_in_library", "Run a NEW open-vocabulary detection scan over stored photos/videos for anything (concrete objects or abstract categories like 'dangerous things', 'all living things', 'menacing people'). Returns every matching detection in the scanned media (new and previously found). Slower than search; results are saved and become searchable.",
           {"query": S, "media_ids": {"type": "array", "items": S, "description": "restrict to these media (default: the most recent media)"},
            "limit_media": {**I, "description": "max number of most recent media to scan when media_ids is not given (default 50, max 500)"},
            "verify": {"type": "boolean", "description": "VLM-verify every candidate (default: automatic)"}}, ["query"])
        async def find_in_library(query: str, media_ids: Optional[list[str]] = None, limit_media: int = 50, verify: Optional[bool] = None) -> ToolOutput:
            from ..pipeline.analyzer import AnalyzeOptions

            if media_ids:
                known = {m["id"] for m in await db.get_media_many(media_ids)}
                unknown = [i for i in media_ids if i not in known]
                if unknown:
                    return _fail(f"unknown media id(s): {', '.join(unknown)}")
                ids = list(dict.fromkeys(media_ids))
            else:
                ids = await db.all_media_ids(limit=max(1, min(limit_media, 500)))
            if not ids:
                return _fail("the library is empty")
            opts = AnalyzeOptions(query=query, verify=verify if verify is not None else "auto", describe="none")
            job = self.ctx.jobs.submit_scan(ids, opts)
            try:
                await self.ctx.jobs.wait(job, timeout=900)
            except asyncio.TimeoutError:
                return ToolOutput(f"scan still running (job {job.id})", {"job_id": job.id, "status": job.status, "progress": job.progress}, job_id=job.id)
            if job.status != "done":
                return ToolOutput(f"scan {job.status}", {"job_id": job.id, "status": job.status, "error": job.error}, job_id=job.id)
            concepts = (job.plan or {}).get("concepts", [])
            ents = await self._concept_hits(ids, concepts)
            hit_media = {e["media_id"] for e in ents}
            media = [serialize.media(m) for m in await db.get_media_many(list(hit_media))]
            data: dict[str, Any] = {
                "plan": [{"concept": c["name"], "prompts": c["prompts"][:12], "verified_by_vlm": c["verify"]} for c in concepts],
                "scanned_media": len(ids),
                **{k: job.summary.get(k, 0) for k in ("accepted", "rejected", "updated")},  # this scan's candidates
                "hits": len(ents),
                "media_with_hits": len(hit_media),
                "found": _named(ents[:60], media),
            }
            return ToolOutput(f"scanned {len(ids)} media, {len(ents)} hit(s)", data, ents, media, job_id=job.id)

        @t("plan_query", "Preview how a query would be expanded into detector prompts and VLM checks (no detection).", {"query": S}, ["query"])
        async def plan_query(query: str) -> ToolOutput:
            from ..pipeline.analyzer import AnalyzeOptions

            plan = await self.ctx.analyzer.make_plan(AnalyzeOptions(query=query))
            return ToolOutput(f"{len(plan.concepts)} concept(s)", plan.to_dict())

        @t("list_media", "List photos/videos in the library, newest first.", {"limit": I, "offset": I, "kind": {"type": "string", "enum": ["image", "video"]}})
        async def list_media(limit: int = 20, offset: int = 0, kind: Optional[str] = None) -> ToolOutput:
            rows = await db.list_media(min(max(limit, 1), 100), max(offset, 0), kind)
            total = await db.count_media(kind)
            media = [serialize.media(r) for r in rows]
            keys = ("id", "kind", "name", "entity_count", "person_count", "query", "duration")
            return ToolOutput(f"{len(media)} of {total} media", {"total": total, "media": [{k: m[k] for k in keys if m.get(k) is not None} for m in media]}, media=media)

        @t("get_media", "Get one photo/video with all its detections (labels, VLM descriptions, linked persons).", {"media_id": S}, ["media_id"])
        async def get_media(media_id: str) -> ToolOutput:
            m = await db.get_media(media_id)
            if m is None:
                return _fail(f"media {media_id} not found")
            ents = [serialize.entity(e) for e in await db.media_entities(media_id)]
            md = serialize.media(m)
            kept = [serialize.compact_entity(e) for e in ents if e["stage"] != "rejected"]
            return ToolOutput(f"{md['name']}: {len(kept)} entities", {"media": {k: md[k] for k in ("id", "kind", "name", "width", "height", "duration")}, "entities": kept}, ents, [md])

        @t("ask_image", "Ask the vision-language model a question about a photo/video (media_id) or about one detection (entity_id), e.g. 'what is happening here?'. Give exactly one of media_id or entity_id.",
           {"question": S, "media_id": S, "entity_id": S, "scope": {"type": "string", "enum": ["crop", "scene"], "description": "for entity questions: the crop only (default), or the whole scene"}}, ["question"])
        async def ask_image(question: str, media_id: Optional[str] = None, entity_id: Optional[str] = None, scope: str = "crop") -> ToolOutput:
            if entity_id:
                rows = await db.get_entities([entity_id])
                if not rows:
                    return _fail(f"entity {entity_id} not found")
                ans = await self.ctx.analyzer.ask_entity(entity_id, question, scope)
                return ToolOutput(ans[:120], {"answer": ans}, [serialize.entity(r) for r in rows])
            if media_id:
                m = await db.get_media(media_id)
                if m is None:
                    return _fail(f"media {media_id} not found")
                ans = await self.ctx.analyzer.ask_media(media_id, question)
                return ToolOutput(ans[:120], {"answer": ans}, media=[serialize.media(m)])
            return _fail("provide media_id or entity_id")

        @t("list_persons", "List identity clusters (people seen across media).",
           {"query": {**S, "description": "filter by name/notes"}, "watchlist": {"type": "boolean", "description": "only watchlisted persons"},
            "order": {"type": "string", "enum": ["recent", "sightings"], "description": "default: sightings (most seen first)"}, "limit": I})
        async def list_persons(query: Optional[str] = None, watchlist: bool = False, order: str = "sightings", limit: int = 20) -> ToolOutput:
            rows, total = await db.list_persons(watchlist, query or None, min(max(limit, 1), 100), 0, order)
            ps = [serialize.person(r) for r in rows]
            return ToolOutput(f"{len(ps)} of {total} person(s)", {"total": total, "persons": [_compact_person(p) for p in ps]}, persons=ps)

        @t("get_person", "Get a person cluster with its sightings (VLM descriptions per sighting).", {"person_id": S}, ["person_id"])
        async def get_person(person_id: str) -> ToolOutput:
            p = await db.get_person(person_id)
            if p is None:
                return _fail(f"person {person_id} not found")
            sights = [serialize.entity(r) for r in await db.person_sightings(person_id, 100)]
            media = [serialize.media(m) for m in await db.get_media_many(list({e["media_id"] for e in sights}))]
            pd = serialize.person(p)
            return ToolOutput(f"{pd['label'] or person_id}: {len(sights)} sighting(s)", {"person": _compact_person(pd), "sightings": _named(sights[:40], media)}, sights, media, [pd])

        @t("name_person", "Give a person cluster a name/label (and optional notes).", {"person_id": S, "name": S, "notes": S}, ["person_id", "name"])
        async def name_person(person_id: str, name: str, notes: Optional[str] = None) -> ToolOutput:
            if not await db.update_person(person_id, label=name, notes=notes):
                return _fail(f"person {person_id} not found")
            return await person_out(person_id, f"named {name}")

        @t("set_watchlist", "Add (on=true) or remove (on=false) a person from the watchlist.", {"person_id": S, "on": {"type": "boolean"}}, ["person_id", "on"])
        async def set_watchlist(person_id: str, on: bool) -> ToolOutput:
            if not await db.update_person(person_id, is_watchlist=on):
                return _fail(f"person {person_id} not found")
            return await person_out(person_id, "added to watchlist" if on else "removed from watchlist")

        @t("merge_persons", "Merge person `other_id` into `keep_id` (they are the same individual). Irreversible.", {"keep_id": S, "other_id": S}, ["keep_id", "other_id"])
        async def merge_persons(keep_id: str, other_id: str) -> ToolOutput:
            if keep_id == other_id:
                return _fail("keep_id and other_id are the same person")
            missing = [i for i in (keep_id, other_id) if await db.get_person(i) is None]
            if missing:
                return _fail(f"person(s) not found: {', '.join(missing)}")
            if not await self.ctx.identity.merge(keep_id, other_id):  # serialised with identity matching
                return _fail("merge failed")
            return await person_out(keep_id, f"merged {other_id} into {keep_id}")

        @t("recluster_persons", "Re-run identity clustering over all person sightings (named people are kept as anchors).", {"threshold": {"type": "number", "description": "cosine threshold; lower merges more"}})
        async def recluster_persons(threshold: Optional[float] = None) -> ToolOutput:
            space = self.ctx.reid_space()
            if not space:
                return _fail("Re-ID is unavailable (no Re-ID model)")
            res = await self.ctx.identity.recluster(space, threshold)
            rows, _ = await db.list_persons(False, None, 30, 0, "sightings")
            ps = [serialize.person(r) for r in rows]
            return ToolOutput(f"{res['persons']} person(s), {res['changed']} change(s)", {**res, "top_persons": [_compact_person(p) for p in ps]}, persons=ps)

        @t("find_person_sightings", "Find where a person appears (photos/videos and sightings), by person id or by name.", {"person_id": S, "name": S})
        async def find_person_sightings(person_id: Optional[str] = None, name: Optional[str] = None) -> ToolOutput:
            if not person_id and name:
                matches = await db.find_persons_by_label(name) or (await db.list_persons(False, name, 10, 0))[0]
                if not matches:
                    return _fail(f"no person named {name!r}")
                if len(matches) > 1:
                    ps = [serialize.person(r) for r in matches]
                    return ToolOutput(f"{len(ps)} persons match {name!r}", {"error": "ambiguous name, ask the user which one", "candidates": [_compact_person(p) for p in ps]}, persons=ps)
                person_id = matches[0]["id"]
            if not person_id:
                return _fail("provide person_id or name")
            p = await db.get_person(person_id)
            if p is None:
                return _fail(f"person {person_id} not found")
            pd = serialize.person(p)
            sights = [serialize.entity(r) for r in await db.person_sightings(person_id, 200)]
            media = [serialize.media(m) for m in await db.get_media_many(list({e["media_id"] for e in sights}))]
            data = {"person": _compact_person(pd), "media": [{"id": m["id"], "name": m["name"], "kind": m["kind"]} for m in media], "sightings": _named(sights[:50], media)}
            return ToolOutput(f"{len(sights)} sighting(s) in {len(media)} media", data, sights, media, [pd])

        @t("stats", "Library statistics: media/entity/person counts and the count of every detected label.", {})
        async def stats() -> ToolOutput:
            st = await db.stats()
            return ToolOutput(f"{st['media']} media, {st['entities']} entities, {st['persons']} persons", st)

        @t("analyze_paths", "Ingest photos/videos from server-side files or folders (runs the full pipeline).",
           {"paths": {"type": "array", "items": S}, "query": {**S, "description": "optional open-vocabulary query"}, "wait": {"type": "boolean", "description": "wait for completion (default true)"}}, ["paths"])
        async def analyze_paths(paths: list[str], query: Optional[str] = None, wait: bool = True) -> ToolOutput:
            from ..pipeline.analyzer import AnalyzeOptions

            try:
                items = await self.ctx.jobs.collect_paths(paths)
            except PermissionError as e:  # outside OVP_IMPORT_ROOTS, or server-side imports disabled
                return _fail(str(e))
            if not items:
                return _fail("no images/videos at those paths")
            job = self.ctx.jobs.submit_analyze(items, AnalyzeOptions(query=query))
            if wait:
                try:
                    await self.ctx.jobs.wait(job, timeout=1800)
                except asyncio.TimeoutError:
                    pass
            media = [serialize.media(m) for m in await db.get_media_many(job.media_ids)] if job.media_ids else []
            info = {k: v for k, v in job.to_dict().items() if k in ("id", "status", "progress", "summary", "error")}
            return ToolOutput(f"job {job.id}: {job.status}, {len(job.media_ids)}/{len(items)} done", {"job": info}, media=media, job_id=job.id)

        @t("job_status", "Status of a background job.", {"job_id": S}, ["job_id"])
        async def job_status(job_id: str) -> ToolOutput:
            job = self.ctx.jobs.get(job_id)
            if job is None:
                return _fail(f"job {job_id} not found")
            return ToolOutput(job.status, {k: v for k, v in job.to_dict().items() if k in ("id", "status", "progress", "summary", "error", "media_ids")}, job_id=job_id)

    async def _concept_hits(self, media_ids: list[str], concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Accepted detections of a plan's concepts in ``media_ids``, including earlier ones:
        entities tagged with a concept name, or (for unverified concepts) carrying one of its prompts as label."""
        names = {c["name"] for c in concepts}
        labels = {p.casefold() for c in concepts if not c["verify"] for p in c["prompts"]}
        rows = await self.ctx.db.media_entities_many(media_ids)
        return [serialize.entity(r) for r in rows if r["stage"] != "rejected" and (r["concept"] in names or r["label"].casefold() in labels)]

    # ── conversations ─────────────────────────────────────────────────────────
    def get_conversation(self, cid: Optional[str]) -> Conversation:
        """Return conversation ``cid``, or start one (under ``cid`` when given, e.g. after a restart)."""
        if cid and cid in self.conversations:
            return self.conversations[cid]
        conv = Conversation(id=cid or new_id("c"))
        conv.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        self.conversations[conv.id] = conv
        idle = [c for c in self.conversations.values() if not c.lock.locked() and c is not conv]
        while len(self.conversations) > MAX_CONVERSATIONS and idle:
            oldest = min(idle, key=lambda c: c.updated_at)
            idle.remove(oldest)
            self.conversations.pop(oldest.id, None)
        return conv

    def list_conversations(self) -> list[dict[str, Any]]:
        convs = sorted(self.conversations.values(), key=lambda c: c.updated_at, reverse=True)
        return [
            {"id": c.id, "title": c.title, "created_at": c.created_at, "updated_at": c.updated_at, "message_count": sum(m["role"] in ("user", "assistant") for m in c.messages)}
            for c in convs
        ]

    def start_turn(self, conv: Conversation, text: str, llm: ChatModel, emit: Emit) -> asyncio.Task:
        """Run one turn as an independent task (it survives client disconnects)."""
        task = asyncio.create_task(self.run(conv, text, llm, emit))
        self._turns.add(task)
        task.add_done_callback(self._turns.discard)
        return task

    async def run(self, conv: Conversation, text: str, llm: ChatModel, emit: Emit, max_steps: int = 12) -> None:
        async with conv.lock:
            _close_dangling_tool_calls(conv.messages)
            conv.messages.append({"role": "user", "content": text})
            conv.updated_at = time.time()
            specs = [spec for spec, _ in self.tools.values()]
            try:
                for _ in range(max_steps):
                    resp = await llm.chat(_trim(conv.messages), specs, max_tokens=1500)
                    conv.messages.append({"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls})
                    if resp.text:
                        await emit("message", text=resp.text)
                    if not resp.tool_calls:
                        if not resp.text:
                            await emit("error", message=f"{llm.spec} returned an empty reply")
                        return
                    outs = await asyncio.gather(*(self._call(tc, emit) for tc in resp.tool_calls))
                    conv.messages.extend(outs)
                # Tools stay declared (providers such as Anthropic reject tool blocks without them);
                # any further tool calls are ignored.
                final = await llm.chat(_trim(conv.messages) + [{"role": "user", "content": SUMMARIZE_PROMPT}], specs, max_tokens=800)
                summary = final.text or "I ran out of tool steps before reaching an answer."
                conv.messages.append({"role": "assistant", "content": summary})
                await emit("message", text=summary)
            finally:
                conv.updated_at = time.time()

    async def _call(self, tc: ToolCall, emit: Emit) -> dict[str, Any]:
        """Run a tool call and return its tool message for the transcript."""
        await emit("tool_call", id=tc.id, name=tc.name, arguments=tc.arguments)
        t0 = time.perf_counter()
        out = await self._invoke(tc)
        await emit(
            "tool_result", id=tc.id, name=tc.name, summary=out.summary, elapsed=round(time.perf_counter() - t0, 2),
            entities=out.entities[:60], media=out.media[:40], persons=out.persons[:40], job_id=out.job_id,
        )
        payload = json.dumps(out.data, default=str)
        if len(payload) > MAX_TOOL_RESULT_CHARS:
            payload = payload[:MAX_TOOL_RESULT_CHARS] + "…(truncated)"
        return {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": payload, "is_error": out.is_error}

    async def _invoke(self, tc: ToolCall) -> ToolOutput:
        entry = self.tools.get(tc.name)
        if entry is None:
            return _fail(f"unknown tool {tc.name!r}")
        fn = entry[1]
        if "_raw" in (tc.arguments or {}):
            return _fail("arguments were not valid JSON")
        args = {k: v for k, v in (tc.arguments or {}).items() if v is not None}
        try:
            inspect.signature(fn).bind(**args)
        except TypeError as e:
            return _fail(f"bad arguments: {e}")
        try:
            return await fn(**args)
        except KeyError as e:
            return _fail(f"not found: {e.args[0] if e.args else e}")
        except Exception as e:
            log.exception("tool %s failed", tc.name)
            return _fail(f"{type(e).__name__}: {e}")

    def transcript(self, conv: Conversation) -> list[dict[str, Any]]:
        out = []
        for m in conv.messages:
            if m["role"] == "system":
                continue
            if m["role"] == "tool":
                out.append({"role": "tool", "name": m.get("name"), "text": m["content"][:2000]})
            else:
                item: dict[str, Any] = {"role": m["role"], "text": m["content"] if isinstance(m["content"], str) else ""}
                if m.get("tool_calls"):
                    item["tool_calls"] = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m["tool_calls"]]
                out.append(item)
        return out


def _close_dangling_tool_calls(messages: list[dict[str, Any]]) -> None:
    """Give every assistant tool call a tool result (a turn may have been interrupted between
    the two); providers reject transcripts with unanswered tool calls."""
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        out.append(m)
        i += 1
        calls: list[ToolCall] = m.get("tool_calls") or []
        if not calls:
            continue
        answered: set[str] = set()
        while i < len(messages) and messages[i]["role"] == "tool":
            answered.add(messages[i]["tool_call_id"])
            out.append(messages[i])
            i += 1
        for tc in calls:
            if tc.id not in answered:
                out.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps({"error": "interrupted before the tool finished"}), "is_error": True})
    messages[:] = out


def _trim(messages: list[dict[str, Any]], max_chars: int = HISTORY_BUDGET_CHARS) -> list[dict[str, Any]]:
    """The system prompt plus the longest run of whole recent turns within a character budget.

    The window always starts at a user message (so tool results stay with their assistant tool
    call, and providers that require a leading user turn are satisfied) and always includes the
    current turn, even when that alone exceeds the budget."""
    system, rest = messages[0], messages[1:]
    turn_starts = [i for i, m in enumerate(rest) if m["role"] == "user"]
    if not turn_starts:
        return messages
    start = turn_starts[-1]
    size = sum(_size(m) for m in rest[start:])
    for prev in reversed(turn_starts[:-1]):
        size += sum(_size(m) for m in rest[prev:start])
        if size > max_chars:
            break
        start = prev
    return [system] + rest[start:]


def _size(m: dict[str, Any]) -> int:
    content = m.get("content")
    n = len(content) if isinstance(content, str) else 500
    return n + sum(len(json.dumps(tc.arguments, default=str)) for tc in m.get("tool_calls") or [])


def _uniq(items: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for it in items:
        seen.setdefault(it["id"], it)
    return list(seen.values())


def _named(entities: list[dict], media: list[dict]) -> list[dict]:
    """Compact entities with their photo/video file name, so answers can name the file."""
    names = {m["id"]: m["name"] for m in media}
    return [{**serialize.compact_entity(e), "media_name": names.get(e["media_id"])} for e in entities]


def _compact_person(p: dict) -> dict:
    return {k: p[k] for k in ("id", "label", "is_watchlist", "notes", "sighting_count", "media_count") if p.get(k) not in (None, "")}
