"""Analyst agent: the SSE chat contract, tool dispatch/errors, transcript validity and isolation.

The LLM is scripted: each test lists the replies it returns, and every request it receives is
recorded so the transcript sent to the provider can be checked.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable, Union

import httpx
import pytest

from ovp.agent.agent import Agent, Conversation, _close_dangling_tool_calls, _trim
from ovp.api.app import app
from ovp.models.llm import ChatResponse, ToolCall
from ovp.storage.db import Database

from .seed import add_entities, add_media

Reply = Union[ChatResponse, Callable[[list[dict]], Any]]


class ScriptedLLM:
    supports_tools = True
    spec = "fake:scripted"

    def __init__(self, *replies: Reply):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    async def chat(self, messages: list[dict], tools: Any = None, **_: Any) -> ChatResponse:
        self.requests.append(list(messages))
        reply = self.replies.pop(0)
        return await reply(messages) if callable(reply) else reply


def call(name: str, call_id: str = "call_1", **arguments: Any) -> ChatResponse:
    return ChatResponse(text="", tool_calls=[ToolCall(call_id, name, arguments)])


def say(text: str) -> ChatResponse:
    return ChatResponse(text=text)


@pytest.fixture
async def client(app_ctx: Any) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def chat(client: httpx.AsyncClient, message: str, conversation_id: str | None = None) -> list[dict[str, Any]]:
    r = await client.post("/api/agent/chat", json={"message": message, "conversation_id": conversation_id})
    assert r.status_code == 200, r.text
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def assert_valid_transcript(messages: list[dict]) -> None:
    """Every assistant tool call is answered by tool messages directly after it."""
    for i, m in enumerate(messages):
        if m["role"] == "tool":
            j = i
            while messages[j]["role"] == "tool":
                j -= 1
            assert messages[j]["role"] == "assistant" and m["tool_call_id"] in {tc.id for tc in messages[j]["tool_calls"]}
        if m["role"] == "assistant" and m.get("tool_calls"):
            answered = {x["tool_call_id"] for x in messages[i + 1 : i + 1 + len(m["tool_calls"])] if x["role"] == "tool"}
            assert answered == {tc.id for tc in m["tool_calls"]}


async def test_chat_streams_documented_events_and_keeps_history(client: httpx.AsyncClient, app_ctx: Any, db: Database) -> None:
    media_id = await add_media(db, name="street.jpg")
    await add_entities(db, media_id, [None, None], label="bus")
    llm = ScriptedLLM(call("stats"), say("There are 2 buses."), say("Still 2."))
    app_ctx.registry.models["llm"] = llm

    events = await chat(client, "how many buses?")
    assert [e["type"] for e in events] == ["conversation", "tool_call", "tool_result", "message", "done"]
    cid = events[0]["conversation_id"]
    assert events[1] == {"type": "tool_call", "id": "call_1", "name": "stats", "arguments": {}}
    result = events[2]
    assert result["summary"].startswith("1 media, 2 entities") and {"entities", "media", "persons", "job_id"} <= result.keys()
    assert events[3]["text"] == "There are 2 buses."

    await chat(client, "and now?", cid)
    second = llm.requests[-1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert json.loads(second[3]["content"])["labels"] == [{"label": "bus", "count": 2}]

    listing = (await client.get("/api/agent/conversations")).json()["conversations"]
    assert listing[0]["id"] == cid and listing[0]["title"] == "how many buses?" and listing[0]["message_count"] == 5
    transcript = (await client.get(f"/api/agent/conversations/{cid}")).json()["messages"]
    assert [m["role"] for m in transcript] == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    assert transcript[1]["tool_calls"] == [{"id": "call_1", "name": "stats", "arguments": {}}]
    assert (await client.get("/api/agent/conversations/c_missing")).status_code == 404


async def test_tool_errors_are_reported_to_the_model(client: httpx.AsyncClient, app_ctx: Any) -> None:
    llm = ScriptedLLM(
        ChatResponse("", [
            ToolCall("a", "no_such_tool", {}),
            ToolCall("b", "get_media", {"media": "m_x"}),  # wrong argument name
            ToolCall("c", "get_media", {"_raw": "{not json"}),
            ToolCall("d", "get_media", {"media_id": "m_missing"}),
            ToolCall("e", "name_person", {"person_id": "p_missing", "name": "Alice"}),
            ToolCall("f", "merge_persons", {"keep_id": "p_1", "other_id": "p_1"}),
            ToolCall("g", "ask_image", {"question": "what?"}),
        ]),
        say("Those ids do not exist."),
    )
    app_ctx.registry.models["llm"] = llm
    events = await chat(client, "check things")
    summaries = {e["id"]: e["summary"] for e in events if e["type"] == "tool_result"}
    assert summaries == {
        "a": "unknown tool 'no_such_tool'",
        "b": "bad arguments: missing a required argument: 'media_id'",
        "c": "arguments were not valid JSON",
        "d": "media m_missing not found",
        "e": "person p_missing not found",
        "f": "keep_id and other_id are the same person",
        "g": "provide media_id or entity_id",
    }
    tool_msgs = [m for m in llm.requests[-1] if m["role"] == "tool"]
    assert all("error" in json.loads(m["content"]) for m in tool_msgs) and len(tool_msgs) == 7
    assert events[-2:] == [{"type": "message", "text": "Those ids do not exist."}, {"type": "done"}]


async def test_provider_error_becomes_error_event(client: httpx.AsyncClient, app_ctx: Any) -> None:
    async def boom(_: list[dict]) -> ChatResponse:
        raise RuntimeError("provider down")

    app_ctx.registry.models["llm"] = ScriptedLLM(boom)
    events = await chat(client, "hi")
    assert [e["type"] for e in events] == ["conversation", "error", "done"]
    assert events[1]["message"] == "provider down"


async def test_unavailable_llm_spec_is_a_client_error(client: httpx.AsyncClient, app_ctx: Any) -> None:
    app_ctx.registry.models["llm"] = None
    r = await client.post("/api/agent/chat", json={"message": "hi", "llm": "bogus:model"})
    assert r.status_code == 400 and "bogus:model" in r.json()["detail"]
    assert (await client.post("/api/agent/chat", json={"message": "hi"})).status_code == 503


async def test_name_and_watchlist_return_the_person(app_ctx: Any, db: Database) -> None:
    media_id = await add_media(db)
    entity_id = (await add_entities(db, media_id, [None]))[0]
    person_id = await db.create_person("reid:osnet:fake", None, entity_id)
    await db.link_entity(entity_id, person_id, "auto", 0.9)
    tools = app_ctx.agent.tools
    named = await tools["name_person"][1](person_id=person_id, name="Alice")
    assert named.data == {"person": {"id": person_id, "label": "Alice", "is_watchlist": False, "sighting_count": 1, "media_count": 1}}
    watched = await tools["set_watchlist"][1](person_id=person_id, on=True)
    assert watched.persons[0]["is_watchlist"] is True
    found = await tools["find_person_sightings"][1](name="alice")
    assert found.summary == "1 sighting(s) in 1 media" and found.persons[0]["id"] == person_id


async def test_concept_hits_include_earlier_detections(app_ctx: Any, db: Database) -> None:
    media_id = await add_media(db)
    tie, = await add_entities(db, media_id, [None], label="tie", extra={"concept": "ties"})
    menacing, = await add_entities(db, media_id, [None], extra={"concept": "menacing people", "verified": True})
    await add_entities(db, media_id, [None], extra={"stage": "rejected"})
    await add_entities(db, media_id, [None], label="dog")
    concepts = [
        {"name": "tie", "prompts": ["tie"], "verify": False},
        {"name": "menacing people", "prompts": ["person"], "verify": True},
    ]
    hits = await app_ctx.agent._concept_hits([media_id], concepts)
    assert {e["id"] for e in hits} == {tie, menacing}


async def test_concurrent_conversations_are_isolated(app_ctx: Any) -> None:
    agent: Agent = app_ctx.agent

    def echo_after(delay: float) -> Callable[[list[dict]], Any]:
        async def reply(messages: list[dict]) -> ChatResponse:
            await asyncio.sleep(delay)
            return say("seen: " + " | ".join(m["content"] for m in messages if m["role"] == "user"))

        return reply

    a, b = agent.get_conversation(None), agent.get_conversation(None)
    outputs: dict[str, list[str]] = {a.id: [], b.id: []}

    def collector(cid: str) -> Callable[..., Any]:
        async def emit(type: str, **data: Any) -> None:
            if type == "message":
                outputs[cid].append(data["text"])

        return emit

    llm_a, llm_b = ScriptedLLM(echo_after(0.05), echo_after(0)), ScriptedLLM(echo_after(0))
    await asyncio.gather(
        agent.start_turn(a, "alpha", llm_a, collector(a.id)),  # type: ignore[arg-type]
        agent.start_turn(a, "alpha 2", llm_a, collector(a.id)),  # type: ignore[arg-type]
        agent.start_turn(b, "beta", llm_b, collector(b.id)),  # type: ignore[arg-type]
    )
    assert outputs[a.id] == ["seen: alpha", "seen: alpha | alpha 2"]  # same conversation: serialized
    assert outputs[b.id] == ["seen: beta"]


async def test_interrupted_turn_leaves_a_valid_transcript(app_ctx: Any) -> None:
    agent: Agent = app_ctx.agent
    started = asyncio.Event()

    async def slow_tool(**_: Any) -> Any:
        started.set()
        await asyncio.sleep(30)

    spec = agent.tools["stats"][0]
    agent.tools["stats"] = (spec, slow_tool)
    conv = agent.get_conversation(None)

    async def emit(type: str, **_: Any) -> None:
        return None

    llm = ScriptedLLM(call("stats"), say("ok"))
    task = agent.start_turn(conv, "stats please", llm, emit)  # type: ignore[arg-type]
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await agent.run(conv, "again", llm, emit)  # type: ignore[arg-type]
    sent = llm.requests[-1]
    assert_valid_transcript(sent)
    assert json.loads(sent[3]["content"]) == {"error": "interrupted before the tool finished"}


def _turn(i: int, size: int) -> list[dict]:
    return [
        {"role": "user", "content": f"q{i}"},
        {"role": "assistant", "content": "", "tool_calls": [ToolCall(f"c{i}", "stats", {})]},
        {"role": "tool", "tool_call_id": f"c{i}", "name": "stats", "content": "x" * size},
        {"role": "assistant", "content": f"a{i}"},
    ]


def test_trim_keeps_whole_turns_starting_with_a_user_message() -> None:
    messages = [{"role": "system", "content": "sys"}] + [m for i in range(5) for m in _turn(i, 400)]
    trimmed = _trim(messages, max_chars=1000)
    assert trimmed[0]["content"] == "sys" and trimmed[1]["content"] == "q3"
    assert_valid_transcript(trimmed)
    huge = [{"role": "system", "content": "sys"}] + _turn(0, 10) + _turn(1, 5000)
    assert [m.get("content") for m in _trim(huge, max_chars=1000)][:2] == ["sys", "q1"]  # current turn always kept
    assert _trim(messages, max_chars=10**6) == messages


def test_close_dangling_tool_calls() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [ToolCall("a", "stats", {}), ToolCall("b", "stats", {})]},
        {"role": "tool", "tool_call_id": "a", "name": "stats", "content": "{}"},
        {"role": "user", "content": "next"},
    ]
    _close_dangling_tool_calls(messages)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool", "user"]
    assert messages[4]["tool_call_id"] == "b"
    assert_valid_transcript(messages)


def test_conversation_title() -> None:
    conv = Conversation(id="c_1", messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "find ties"}])
    assert conv.title == "find ties"
