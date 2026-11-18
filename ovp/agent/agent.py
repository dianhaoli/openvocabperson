"""Analyst agent: an LLM with tools over the media library.

The agent can search, run new open-vocabulary scans over stored photos/videos, inspect and
question images with the VLM, cluster / name / merge people, and ingest folders. Tool results
carry full entity/media/person objects for the UI (rendered as galleries) and a compact JSON
summary for the model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .. import serialize
from ..models.llm import ChatModel, ToolCall, ToolSpec
from ..storage.db import new_id, rows_by_id

log = logging.getLogger("ovp.agent")

SYSTEM_PROMPT = """You are OpenVocab Analyst, an assistant that investigates a private library of photos and videos.
Every photo/video has been processed: objects were detected (open-vocabulary), people were described by a vision-language model and grouped into identity clusters ("persons") via Re-ID.

How to work:
- Use tools; never invent entities, people, counts or file names. Prefer `search` first for anything already detected/described.
- If the user asks for something that may not have been detected yet (e.g. "dangerous things", "all living things", a specific object), run `find_in_library`, which re-scans stored media with an open-vocabulary detector (and an LLM-expanded vocabulary + VLM verification for abstract requests).
- For questions about what is happening in a specific image or about a specific detection, use `ask_image`.
- For people: `list_persons` (order by sightings to find who appears most), `get_person`, `name_person`, `set_watchlist`, `merge_persons` (only when the user asks or the evidence is strong — say why), `recluster_persons`.
- Refer to items by their ids (e.g. e_1a2b3c4d5e, m_..., p_...) so the UI can link them. Keep answers concise, use short bullet lists, and state uncertainty honestly (Re-ID matches are probabilistic)."""


@dataclass
class Conversation:
    id: str
    messages: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class ToolOutput:
    summary: str
    data: Any = None  # sent to the model (compact)
    entities: list[dict] = field(default_factory=list)
    media: list[dict] = field(default_factory=list)
    persons: list[dict] = field(default_factory=list)
    job_id: Optional[str] = None


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or [], "additionalProperties": False}


class Agent:
    def __init__(self, ctx: Any):
        """``ctx`` is the app context (db, analyzer, search, identity, jobs, registry, store)."""
        self.ctx = ctx
        self.conversations: dict[str, Conversation] = {}
        self.tools: dict[str, tuple[ToolSpec, Callable[..., Awaitable[ToolOutput]]]] = {}
        self._register()

    # ── tool registry ─────────────────────────────────────────────────────────
    def _tool(self, name: str, description: str, params: dict[str, Any], required: list[str] | None = None):
        def deco(fn: Callable[..., Awaitable[ToolOutput]]):
            self.tools[name] = (ToolSpec(name, description, _obj(params, required)), fn)
            return fn

        return deco

    def _register(self) -> None:
        t = self._tool
        S = {"type": "string"}
        I = {"type": "integer"}

        @t("search", "Hybrid search over detected entities (labels, VLM descriptions, CLIP). Returns matching detections with their media.",
           {"query": {**S, "description": "what to look for, e.g. 'man in red jacket', 'knife', 'people on bikes'"},
            "labels": {"type": "array", "items": S, "description": "optional exact detector labels to restrict to, e.g. ['person']"},
            "media_id": S, "person_id": S, "limit": I}, ["query"])
        async def search(query: str, labels: Optional[list[str]] = None, media_id: Optional[str] = None, person_id: Optional[str] = None, limit: int = 12) -> ToolOutput:
            hits = await self.ctx.search.search(query, None, {"labels": labels, "media_id": media_id, "person_id": person_id}, min(max(limit, 1), 40))
            res = await self.ctx.search.hydrate(hits)
            ents = [r["entity"] for r in res]
            media = _uniq([r["media"] for r in res if r["media"]])
            return ToolOutput(f"{len(ents)} result(s)", {"results": [{**serialize.compact_entity(r["entity"]), "media_name": (r["media"] or {}).get("name"), "score": r["score"]} for r in res]}, ents, media)

        @t("find_in_library", "Run a NEW open-vocabulary detection scan over stored photos/videos for anything (concrete objects or abstract categories like 'dangerous things', 'all living things', 'menacing people'). Slower than search; results are saved and become searchable.",
           {"query": S, "media_ids": {"type": "array", "items": S, "description": "restrict to these media (default: whole library)"},
            "limit_media": {**I, "description": "max number of most recent media to scan (default 50)"},
            "verify": {"type": "boolean", "description": "VLM-verify every candidate (default: automatic)"}}, ["query"])
        async def find_in_library(query: str, media_ids: Optional[list[str]] = None, limit_media: int = 50, verify: Optional[bool] = None) -> ToolOutput:
            from ..pipeline.analyzer import AnalyzeOptions

            opts = AnalyzeOptions(query=query, verify=verify if verify is not None else "auto", describe="none")
            job = self.ctx.jobs.submit_scan(media_ids, opts, limit=None if media_ids else max(1, min(limit_media, 500)))
            try:
                await self.ctx.jobs.wait(job, timeout=900)
            except asyncio.TimeoutError:
                return ToolOutput(f"scan still running (job {job.id})", {"job_id": job.id, "status": job.status, "progress": job.progress}, job_id=job.id)
            concepts = {c["name"] for c in (job.plan or {}).get("concepts", [])}
            ents: list[dict] = []
            if job.media_ids:
                rows = await self.ctx.db.media_entities_many(job.media_ids)
                ents = [serialize.entity(r) for r in rows if r["concept"] in concepts and r["stage"] != "rejected"]
            media = [serialize.media(m) for m in await self.ctx.db.get_media_many(job.media_ids)] if job.media_ids else []
            data = {
                "plan": [{"concept": c["name"], "prompts": c["prompts"][:12], "verified_by_vlm": c["verify"]} for c in (job.plan or {}).get("concepts", [])],
                "scanned": job.summary.get("scanned"),
                "found": [{**serialize.compact_entity(e)} for e in ents[:60]],
                "status": job.status,
            }
            return ToolOutput(f"scanned {job.summary.get('scanned', 0)} media, {len(ents)} hit(s)", data, ents, media, job_id=job.id)

        @t("plan_query", "Preview how a query would be expanded into detector prompts and VLM checks (no detection).", {"query": S}, ["query"])
        async def plan_query(query: str) -> ToolOutput:
            from ..pipeline.analyzer import AnalyzeOptions

            plan = await self.ctx.analyzer.make_plan(AnalyzeOptions(query=query))
            return ToolOutput(f"{len(plan.concepts)} concept(s)", plan.to_dict())

        @t("list_media", "List photos/videos in the library, newest first.", {"limit": I, "offset": I, "kind": {"type": "string", "enum": ["image", "video"]}})
        async def list_media(limit: int = 20, offset: int = 0, kind: Optional[str] = None) -> ToolOutput:
            rows = await self.ctx.db.list_media(min(limit, 100), offset, kind)
            total = await self.ctx.db.count_media(kind)
            media = [serialize.media(r) for r in rows]
            return ToolOutput(f"{len(media)} of {total} media", {"total": total, "media": [{k: m[k] for k in ("id", "kind", "name", "entity_count", "person_count", "query", "duration")} for m in media]}, media=media)

        @t("get_media", "Get one photo/video with all its detections.", {"media_id": S}, ["media_id"])
        async def get_media(media_id: str) -> ToolOutput:
            m = await self.ctx.db.get_media(media_id)
            if m is None:
                return ToolOutput("not found", {"error": "media not found"})
            ents = [serialize.entity(e) for e in await self.ctx.db.media_entities(media_id)]
            md = serialize.media(m)
            return ToolOutput(f"{md['name']}: {len(ents)} entities", {"media": {k: md[k] for k in ("id", "kind", "name", "width", "height", "duration")}, "entities": [serialize.compact_entity(e) for e in ents if e["stage"] != "rejected"]}, ents, [md])

        @t("ask_image", "Ask the vision-language model a question about a photo/video (media_id) or about one detection (entity_id).",
           {"question": S, "media_id": S, "entity_id": S, "scope": {"type": "string", "enum": ["crop", "scene"], "description": "for entity questions: crop only, or the whole scene"}}, ["question"])
        async def ask_image(question: str, media_id: Optional[str] = None, entity_id: Optional[str] = None, scope: str = "crop") -> ToolOutput:
            if entity_id:
                ans = await self.ctx.analyzer.ask_entity(entity_id, question, scope)
                rows = await self.ctx.db.get_entities([entity_id])
                return ToolOutput(ans[:120], {"answer": ans}, [serialize.entity(r) for r in rows])
            if media_id:
                ans = await self.ctx.analyzer.ask_media(media_id, question)
                m = await self.ctx.db.get_media(media_id)
                return ToolOutput(ans[:120], {"answer": ans}, media=[serialize.media(m)] if m else [])
            return ToolOutput("need media_id or entity_id", {"error": "provide media_id or entity_id"})

        @t("list_persons", "List identity clusters (people seen across media).",
           {"query": {**S, "description": "filter by name/notes"}, "watchlist": {"type": "boolean"}, "order": {"type": "string", "enum": ["recent", "sightings"]}, "limit": I})
        async def list_persons(query: Optional[str] = None, watchlist: bool = False, order: str = "sightings", limit: int = 20) -> ToolOutput:
            rows, total = await self.ctx.db.list_persons(watchlist, query, min(limit, 100), 0, order)
            ps = [serialize.person(r) for r in rows]
            return ToolOutput(f"{len(ps)} of {total} person(s)", {"total": total, "persons": [_compact_person(p) for p in ps]}, persons=ps)

        @t("get_person", "Get a person cluster with all sightings.", {"person_id": S}, ["person_id"])
        async def get_person(person_id: str) -> ToolOutput:
            p = await self.ctx.db.get_person(person_id)
            if p is None:
                return ToolOutput("not found", {"error": "person not found"})
            sights = [serialize.entity(r) for r in await self.ctx.db.person_sightings(person_id, 100)]
            pd = serialize.person(p)
            return ToolOutput(f"{pd['label'] or person_id}: {len(sights)} sighting(s)", {"person": _compact_person(pd), "sightings": [serialize.compact_entity(e) for e in sights[:40]]}, sights, persons=[pd])

        @t("name_person", "Give a person cluster a name/label (and optional notes).", {"person_id": S, "name": S, "notes": S}, ["person_id", "name"])
        async def name_person(person_id: str, name: str, notes: Optional[str] = None) -> ToolOutput:
            ok = await self.ctx.db.update_person(person_id, label=name, notes=notes)
            p = await self.ctx.db.get_person(person_id)
            return ToolOutput("named" if ok else "not found", {"ok": ok}, persons=[serialize.person(p)] if p else [])

        @t("set_watchlist", "Add or remove a person from the watchlist.", {"person_id": S, "on": {"type": "boolean"}}, ["person_id", "on"])
        async def set_watchlist(person_id: str, on: bool) -> ToolOutput:
            ok = await self.ctx.db.update_person(person_id, is_watchlist=on)
            p = await self.ctx.db.get_person(person_id)
            return ToolOutput("updated" if ok else "not found", {"ok": ok}, persons=[serialize.person(p)] if p else [])

        @t("merge_persons", "Merge person `other_id` into `keep_id` (same individual).", {"keep_id": S, "other_id": S}, ["keep_id", "other_id"])
        async def merge_persons(keep_id: str, other_id: str) -> ToolOutput:
            ok = await self.ctx.db.merge_persons(keep_id, other_id)
            p = await self.ctx.db.get_person(keep_id)
            return ToolOutput("merged" if ok else "merge failed", {"ok": ok}, persons=[serialize.person(p)] if p else [])

        @t("recluster_persons", "Re-run identity clustering over all person sightings (named people are kept as anchors).", {"threshold": {"type": "number", "description": "cosine threshold; lower merges more"}})
        async def recluster_persons(threshold: Optional[float] = None) -> ToolOutput:
            space = self.ctx.reid_space()
            if not space:
                return ToolOutput("Re-ID unavailable", {"error": "no Re-ID model"})
            res = await self.ctx.identity.recluster(space, threshold)
            rows, _ = await self.ctx.db.list_persons(False, None, 30, 0, "sightings")
            ps = [serialize.person(r) for r in rows]
            return ToolOutput(f"{res['persons']} person(s), {res['changed']} change(s)", {**res, "top_persons": [_compact_person(p) for p in ps]}, persons=ps)

        @t("find_person_sightings", "Find where a person appears, by person id or by name.", {"person_id": S, "name": S})
        async def find_person_sightings(person_id: Optional[str] = None, name: Optional[str] = None) -> ToolOutput:
            if not person_id and name:
                rows, _ = await self.ctx.db.list_persons(False, name, 5, 0)
                if not rows:
                    return ToolOutput("no such person", {"error": f"no person named {name!r}"})
                person_id = rows[0]["id"]
            if not person_id:
                return ToolOutput("need person_id or name", {"error": "provide person_id or name"})
            sights = [serialize.entity(r) for r in await self.ctx.db.person_sightings(person_id, 200)]
            media = [serialize.media(m) for m in await self.ctx.db.get_media_many(list({e["media_id"] for e in sights}))]
            return ToolOutput(f"{len(sights)} sighting(s) in {len(media)} media", {"person_id": person_id, "media": [{"id": m["id"], "name": m["name"], "kind": m["kind"]} for m in media], "sightings": [serialize.compact_entity(e) for e in sights[:50]]}, sights, media)

        @t("stats", "Library statistics: counts and most common labels.", {})
        async def stats() -> ToolOutput:
            st = await self.ctx.db.stats()
            return ToolOutput(f"{st['media']} media, {st['entities']} entities, {st['persons']} persons", st)

        @t("analyze_paths", "Ingest photos/videos from server-side files or folders (runs the full pipeline).",
           {"paths": {"type": "array", "items": S}, "query": {**S, "description": "optional open-vocabulary query"}, "wait": {"type": "boolean"}}, ["paths"])
        async def analyze_paths(paths: list[str], query: Optional[str] = None, wait: bool = True) -> ToolOutput:
            from ..pipeline.analyzer import AnalyzeOptions

            items = self.ctx.jobs.collect_paths(paths)
            if not items:
                return ToolOutput("no media found", {"error": "no images/videos at those paths"})
            job = self.ctx.jobs.submit_analyze(items, AnalyzeOptions(query=query))
            if wait:
                try:
                    await self.ctx.jobs.wait(job, timeout=1800)
                except asyncio.TimeoutError:
                    pass
            media = [serialize.media(m) for m in await self.ctx.db.get_media_many(job.media_ids)] if job.media_ids else []
            return ToolOutput(f"job {job.id}: {job.status}, {len(job.media_ids)}/{len(items)} done", {"job": {k: v for k, v in job.to_dict().items() if k in ("id", "status", "progress", "summary", "error")}}, media=media, job_id=job.id)

        @t("job_status", "Status of a background job.", {"job_id": S}, ["job_id"])
        async def job_status(job_id: str) -> ToolOutput:
            job = self.ctx.jobs.get(job_id)
            if job is None:
                return ToolOutput("not found", {"error": "job not found"})
            return ToolOutput(job.status, {k: v for k, v in job.to_dict().items() if k in ("id", "status", "progress", "summary", "error", "media_ids")}, job_id=job_id)

    # ── conversation loop ─────────────────────────────────────────────────────
    def get_conversation(self, cid: Optional[str]) -> Conversation:
        if cid and cid in self.conversations:
            return self.conversations[cid]
        conv = Conversation(id=cid or new_id("c"))
        conv.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        self.conversations[conv.id] = conv
        if len(self.conversations) > 200:
            oldest = min(self.conversations.values(), key=lambda c: c.created_at)
            self.conversations.pop(oldest.id, None)
        return conv

    async def run(self, conv: Conversation, text: str, llm: ChatModel, emit: Callable[..., Awaitable[None]], max_steps: int = 12) -> None:
        async with conv.lock:
            conv.messages.append({"role": "user", "content": text})
            specs = [spec for spec, _ in self.tools.values()]
            for step in range(max_steps):
                resp = await llm.chat(_trim(conv.messages), specs, max_tokens=1500)
                conv.messages.append({"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls})
                if resp.text:
                    await emit("message", text=resp.text)
                if not resp.tool_calls:
                    return
                outs = await asyncio.gather(*(self._call(tc, emit) for tc in resp.tool_calls))
                for tc, out in zip(resp.tool_calls, outs):
                    conv.messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": out})
            final = await llm.chat(_trim(conv.messages) + [{"role": "user", "content": "Summarize what you found so far for the user."}], None, max_tokens=800)
            conv.messages.append({"role": "assistant", "content": final.text})
            await emit("message", text=final.text)

    async def _call(self, tc: ToolCall, emit: Callable[..., Awaitable[None]]) -> str:
        await emit("tool_call", id=tc.id, name=tc.name, arguments=tc.arguments)
        entry = self.tools.get(tc.name)
        t0 = time.perf_counter()
        if entry is None:
            out = ToolOutput("unknown tool", {"error": f"unknown tool {tc.name}"})
        else:
            try:
                args = {k: v for k, v in (tc.arguments or {}).items() if v is not None and not k.startswith("_")}
                out = await entry[1](**args)
            except TypeError as e:
                out = ToolOutput("bad arguments", {"error": f"bad arguments: {e}"})
            except Exception as e:
                log.exception("tool %s failed", tc.name)
                out = ToolOutput(f"error: {e}", {"error": str(e)})
        await emit(
            "tool_result", id=tc.id, name=tc.name, summary=out.summary, elapsed=round(time.perf_counter() - t0, 2),
            entities=out.entities[:60], media=out.media[:40], persons=out.persons[:40], job_id=out.job_id,
        )
        payload = json.dumps(out.data, default=str)
        return payload[:14000] + ("…(truncated)" if len(payload) > 14000 else "")

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


def _trim(messages: list[dict], max_chars: int = 120_000) -> list[dict]:
    """Keep the system prompt and the most recent messages within a character budget,
    never starting the window on a tool result (it must follow its assistant tool call)."""
    total = 0
    keep: list[dict] = []
    for m in reversed(messages[1:]):
        size = len(m["content"]) if isinstance(m.get("content"), str) else 500
        if total + size > max_chars and keep:
            break
        keep.append(m)
        total += size
    keep.reverse()
    while keep and keep[0]["role"] == "tool":
        keep.pop(0)
    return [messages[0]] + keep


def _uniq(items: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for it in items:
        seen.setdefault(it["id"], it)
    return list(seen.values())


def _compact_person(p: dict) -> dict:
    return {k: p[k] for k in ("id", "label", "is_watchlist", "notes", "sighting_count", "media_count") if p.get(k) not in (None, "")}
