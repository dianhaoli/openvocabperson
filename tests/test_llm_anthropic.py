"""AnthropicModel: neutral messages → Messages API payload, and response parsing (mocked HTTP)."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import httpx
from PIL import Image

from ovp.models.llm import REASONING_BUDGET_FACTOR, AnthropicModel, ChatResponse, ToolCall, ToolSpec, user


def mocked_model(reply: dict[str, Any]) -> tuple[AnthropicModel, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "test-key" and request.headers["anthropic-version"] == "2023-06-01"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    model = AnthropicModel("claude-test", "https://api.anthropic.com", "test-key", 10.0, 2)
    model._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return model, sent


AGENT_HISTORY: list[dict[str, Any]] = [
    {"role": "system", "content": "You are an analyst."},
    {"role": "user", "content": "who is here and what is in m_1?"},
    {"role": "assistant", "content": "Checking.", "tool_calls": [ToolCall("toolu_a", "list_persons", {"limit": 5}), ToolCall("toolu_b", "get_media", {"media_id": "m_1"})]},
    {"role": "tool", "tool_call_id": "toolu_a", "name": "list_persons", "content": '{"total": 0}'},
    {"role": "tool", "tool_call_id": "toolu_b", "name": "get_media", "content": '{"error": "media m_1 not found"}'},
    {"role": "assistant", "content": "", "tool_calls": [ToolCall("toolu_c", "stats", {})]},
    {"role": "tool", "tool_call_id": "toolu_c", "name": "stats", "content": '{"media": 0}'},
    {"role": "assistant", "content": "The library is empty."},
    {"role": "user", "content": "ok, scan it"},
]


async def test_tool_calls_and_results_convert_to_content_blocks() -> None:
    model, sent = mocked_model({
        "content": [
            {"type": "text", "text": "Scanning."},
            {"type": "tool_use", "id": "toolu_d", "name": "find_in_library", "input": {"query": "ties"}},
        ],
        "stop_reason": "tool_use",
    })
    tools = [ToolSpec("find_in_library", "scan", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]
    resp = await model.chat(AGENT_HISTORY, tools, max_tokens=300)

    assert resp == ChatResponse(text="Scanning.", tool_calls=[ToolCall("toolu_d", "find_in_library", {"query": "ties"})], raw=resp.raw)
    payload = sent[0]
    assert payload["system"] == "You are an analyst."
    assert payload["max_tokens"] == 300 * REASONING_BUDGET_FACTOR and payload["model"] == "claude-test"
    assert payload["tools"] == [{"name": "find_in_library", "description": "scan", "input_schema": tools[0].parameters}]
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "who is here and what is in m_1?"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_a", "name": "list_persons", "input": {"limit": 5}},
            {"type": "tool_use", "id": "toolu_b", "name": "get_media", "input": {"media_id": "m_1"}},
        ]},
        {"role": "user", "content": [  # parallel results share one user turn
            {"type": "tool_result", "tool_use_id": "toolu_a", "content": '{"total": 0}'},
            {"type": "tool_result", "tool_use_id": "toolu_b", "content": '{"error": "media m_1 not found"}'},
        ]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_c", "name": "stats", "input": {}}]},  # no empty text block
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_c", "content": '{"media": 0}'}]},
        {"role": "assistant", "content": [{"type": "text", "text": "The library is empty."}]},
        {"role": "user", "content": [{"type": "text", "text": "ok, scan it"}]},
    ]
    await model.aclose()


async def test_images_and_json_mode() -> None:
    model, sent = mocked_model({"content": [{"type": "text", "text": '{"answer": "red"}'}], "stop_reason": "end_turn"})
    img = Image.new("RGB", (2048, 1024), (200, 10, 10))
    resp = await model.chat([user("What colour?", img)], None, json_mode=True)

    assert resp.text == '{"answer": "red"}' and resp.tool_calls == []
    payload = sent[0]
    assert "tools" not in payload
    assert payload["system"].endswith("Respond with a single valid JSON value and nothing else.")
    text_block, image_block = payload["messages"][0]["content"]
    assert text_block == {"type": "text", "text": "What colour?"}
    assert image_block["type"] == "image" and image_block["source"]["type"] == "base64" and image_block["source"]["media_type"] == "image/jpeg"
    decoded = Image.open(io.BytesIO(base64.b64decode(image_block["source"]["data"])))
    assert decoded.format == "JPEG" and max(decoded.size) == 1024  # downscaled before upload
    await model.aclose()


async def test_tool_results_followed_by_a_user_message_in_the_same_turn() -> None:
    """The agent's final 'summarize' prompt follows tool results directly; both are user content."""
    model, sent = mocked_model({"content": [{"type": "text", "text": "Summary."}]})
    history = AGENT_HISTORY[:7] + [{"role": "user", "content": "Summarize."}]
    await model.chat(history, [ToolSpec("stats", "s", {"type": "object", "properties": {}})])
    roles = [m["role"] for m in sent[0]["messages"]]
    assert roles[-2:] == ["assistant", "user"]  # roles alternate: the text joins the tool-result turn
    assert sent[0]["messages"][-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_c", "content": '{"media": 0}'},
        {"type": "text", "text": "Summarize."},
    ]
    await model.aclose()
