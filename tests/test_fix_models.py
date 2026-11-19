"""Model layer hardening: spec validation and caching in the registry, the Anthropic adapter's
tool-result/truncation handling, image encoding off the event loop, Re-ID spaces, the colour
Re-ID descriptor, and the agent's tool error reporting."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image, ImageOps

from ovp.agent.agent import SYSTEM_PROMPT
from ovp.config import Settings
from ovp.models import llm as L
from ovp.models import registry as R
from ovp.models.embedders import ClipEmbedder, ClipReID, ColorReID
from ovp.models.llm import AnthropicModel, OpenAICompatModel, ProviderError, ToolCall, ToolSpec, user
from ovp.models.registry import InvalidSpec, ModelRegistry, ModelUnavailable, normalize_spec

from .test_agent import ScriptedLLM, call, chat, client, say  # noqa: F401  (fixture re-export)

ASSETS = Path("/usr/local/lib/python3.13/site-packages/ultralytics/assets")


# ── spec validation ───────────────────────────────────────────────────────────


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.weights_dir = tmp_path / "weights"
    s.weights_dir.mkdir()
    return s


@pytest.mark.parametrize(
    ("role", "spec", "canonical"),
    [
        ("detector", "yoloe:yoloe-11s-seg.pt", "yoloe:yoloe-11s-seg.pt"),
        ("detector", " YOLOE : yoloe-11s-seg.pt ", "yoloe:yoloe-11s-seg.pt"),
        ("detector", "yoloe:./yoloe-11s-seg.pt", "yoloe:yoloe-11s-seg.pt"),
        ("detector", "yoloe:.//yoloe-11s-seg.pt", "yoloe:yoloe-11s-seg.pt"),
        ("detector", "yoloe:././yoloe-11s-seg.pt", "yoloe:yoloe-11s-seg.pt"),
        ("detector", "yoloe", "yoloe:yoloe-11l-seg.pt"),
        ("detector", "owlv2:google/owlv2-base-patch16-ensemble", "owlv2:google/owlv2-base-patch16-ensemble"),
        ("detector", "vlm:OpenAI:gpt-4.1", "vlm:openai:gpt-4.1"),
        ("segmenter", "mobilesam", "sam:mobile_sam.pt"),
        ("segmenter", "detector", "detector"),
        ("vlm", "qwen:Qwen/Qwen2.5-VL-3B-Instruct", "qwen:Qwen/Qwen2.5-VL-3B-Instruct"),
        ("vlm", "ollama:qwen2.5vl:7b", "ollama:qwen2.5vl:7b"),
        ("llm", "openrouter:anthropic/claude-sonnet-4", "openrouter:anthropic/claude-sonnet-4"),
        ("embedder", "clip", "clip:ViT-B-32"),
        ("text_embedder", "st:sentence-transformers/all-MiniLM-L6-v2", "st:sentence-transformers/all-MiniLM-L6-v2"),
        ("reid", "osnet", "osnet:osnet_x1_0"),
        ("reid", "CLIP", "clip"),
        ("reid", "none", "none"),
    ],
)
def test_specs_are_normalised(settings: Settings, role: str, spec: str, canonical: str) -> None:
    assert normalize_spec(role, spec, settings) == canonical


@pytest.mark.parametrize(
    ("role", "spec"),
    [
        ("detector", "yolo:/etc/passwd"),
        ("detector", "yolo:../outside.pt"),
        ("detector", "yoloe:weights/../../x.pt"),
        ("detector", "owlv2:org/../x"),
        ("detector", "pickle:x.pt"),
        ("detector", "vlm:evil:model"),
        ("segmenter", "sam:/tmp/x.pt"),
        ("vlm", "qwen:/root/model"),
        ("vlm", "openai:gpt 4"),
        ("embedder", "clip:../ViT"),
        ("embedder", "siglip:x"),
        ("text_embedder", "st:/root/st-model"),
        ("reid", "osnet:resnet50"),
        ("reid", "magic"),
    ],
)
def test_unsafe_or_unknown_specs_are_rejected(settings: Settings, role: str, spec: str) -> None:
    with pytest.raises(InvalidSpec):
        normalize_spec(role, spec, settings)


def test_local_weights_only_inside_weights_dir_or_cwd(settings: Settings, tmp_path: Path) -> None:
    inside = settings.weights_dir / "sub" / "custom.pt"
    assert normalize_spec("detector", f"yolo:{inside}", settings) == f"yolo:{inside}"
    assert normalize_spec("detector", f"yolo:{settings.weights_dir / 'custom.pt'}", settings) == "yolo:custom.pt"
    assert normalize_spec("detector", f"yolo:{Path.cwd() / 'yolo11l.pt'}", settings) == "yolo:yolo11l.pt"
    with pytest.raises(InvalidSpec):
        normalize_spec("detector", f"yolo:{tmp_path / 'custom.pt'}", settings)
    assert issubclass(InvalidSpec, ModelUnavailable)  # API routes report it as a client error


# ── registry cache ────────────────────────────────────────────────────────────


class Loaded:
    def __init__(self, spec: str):
        self.spec = spec
        self.unloaded = 0

    def unload(self) -> None:
        self.unloaded += 1


def counting_registry(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> tuple[ModelRegistry, list[str]]:
    reg = ModelRegistry(settings)
    built: list[str] = []

    def build(role: str, spec: str) -> Loaded:
        if "broken" in spec:
            raise RuntimeError(f"cannot load {spec}")
        built.append(spec)
        return Loaded(spec)

    monkeypatch.setattr(reg, "_build", build)
    return reg, built


def test_spellings_share_one_instance_and_bad_specs_build_nothing(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    reg, built = counting_registry(settings, monkeypatch)
    a = reg.get("detector", "yoloe:./yoloe-11s-seg.pt")
    assert reg.get("detector", "yoloe:.//yoloe-11s-seg.pt") is a
    assert reg.get("detector", "YOLOE:yoloe-11s-seg.pt") is a
    assert built == ["yoloe:yoloe-11s-seg.pt"]
    with pytest.raises(InvalidSpec):
        reg.get("detector", "yolo:/etc/model.pt")
    assert built == ["yoloe:yoloe-11s-seg.pt"] and not reg.errors


def test_default_spec_from_environment_is_trusted(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    settings.detector = "yolo:/opt/models/custom.pt"  # an operator's choice, outside the allowed folders
    reg, built = counting_registry(settings, monkeypatch)
    assert reg.defaults["detector"] == "yolo:/opt/models/custom.pt"
    assert reg.get("detector").spec == "yolo:/opt/models/custom.pt"
    assert reg.get("detector", "yolo:/opt/models/custom.pt") is reg.get("detector")
    assert built == ["yolo:/opt/models/custom.pt"]


def test_extra_models_are_evicted_lru_and_unloaded(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    reg, _ = counting_registry(settings, monkeypatch)
    default = reg.get("embedder", "clip:ViT-B-32")
    reg.defaults["embedder"] = "clip:ViT-B-32"
    extras = [reg.get("detector", f"yolo:m{i}.pt") for i in range(R.MAX_EXTRA_MODELS)]
    reg.get("detector", "yolo:m0.pt")  # used again: now the most recent
    newest = reg.get("detector", "yolo:new.pt")
    cached = {spec for _, spec in reg._cache}
    assert "yolo:m1.pt" not in cached and extras[1].unloaded == 1  # least recently used goes first
    assert {"yolo:m0.pt", "yolo:new.pt", "clip:ViT-B-32"} <= cached and extras[0].unloaded == 0
    assert default.unloaded == 0 and newest.unloaded == 0
    assert len(reg._cache) == R.MAX_EXTRA_MODELS + 1


def test_errors_are_bounded(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    reg, _ = counting_registry(settings, monkeypatch)
    for i in range(R.MAX_ERRORS + 10):
        assert reg.get("detector", f"yolo:broken{i}.pt") is None
    assert len(reg.errors) == R.MAX_ERRORS and "detector|yolo:broken0.pt" not in reg.errors


async def test_switch_defaults_rejects_invalid_specs_atomically(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    reg, built = counting_registry(settings, monkeypatch)
    before = dict(reg.defaults)
    with pytest.raises(InvalidSpec):
        await reg.switch_defaults({"segmenter": "sam:mobile_sam.pt", "detector": "yolo:/etc/x.pt"})
    assert reg.defaults == before and built == []


def test_catalog_cpu_note_has_no_leading_space(settings: Settings) -> None:
    reg = ModelRegistry(settings)
    notes = [e["note"] for e in reg.catalog()["vlm"] if e["local"] and e["spec"].startswith("qwen:")]
    assert notes and all(n and not n.startswith(" ") and "CPU: very slow" in n for n in notes)


# ── Re-ID ─────────────────────────────────────────────────────────────────────


def test_clip_reid_space_names_the_clip_model() -> None:
    assert ClipReID(ClipEmbedder("ViT-L-14")).space == "reid:clip:ViT-L-14"
    assert ClipReID(ClipEmbedder("ViT-B-32")).space != ClipReID(ClipEmbedder("ViT-L-14")).space


def person_crops() -> dict[str, Image.Image]:
    """Detector boxes of the two men of zidane.jpg (both in dark suits) and a man of bus.jpg."""
    zidane = Image.open(ASSETS / "zidane.jpg").convert("RGB")
    bus = Image.open(ASSETS / "bus.jpg").convert("RGB")
    return {
        "zidane": zidane.crop((128, 203, 1106, 716)),
        "ancelotti": zidane.crop((750, 43, 1148, 716)),
        "bus_man": bus.crop((53, 400, 244, 900)),
    }


@pytest.mark.skipif(not ASSETS.exists(), reason="ultralytics sample images missing")
async def test_colour_reid_links_rescaled_mirrored_crops_and_separates_people() -> None:
    crops = person_crops()
    z = crops["zidane"]
    # the same person as a video frame would show him: smaller, mirrored, recompressed
    small = ImageOps.mirror(z.resize((z.width // 3, z.height // 3)))
    reid = ColorReID()
    vecs = await reid.embed([z, small, crops["ancelotti"], crops["bus_man"]])
    assert vecs.shape == (4, ColorReID.dim) and reid.space == "reid:color:v2"
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    same, other_suit, other_coat = float(vecs[0] @ vecs[1]), float(vecs[0] @ vecs[2]), float(vecs[0] @ vecs[3])
    assert same > 0.95
    assert other_suit < 0.88 and other_coat < 0.5  # dark suits no longer look alike


# ── Anthropic adapter ─────────────────────────────────────────────────────────


def anthropic(*replies: dict[str, Any]) -> tuple[AnthropicModel, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    model = AnthropicModel("claude-test", "https://api.anthropic.com", "k", 10.0, 2)
    model._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return model, sent


HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "find m_9"},
    {"role": "assistant", "content": "", "tool_calls": [ToolCall("t1", "get_media", {"media_id": "m_9"})]},
    {"role": "tool", "tool_call_id": "t1", "name": "get_media", "content": '{"error": "media m_9 not found"}', "is_error": True},
]
TOOLS = [ToolSpec("get_media", "g", {"type": "object", "properties": {}})]
TRUNCATED = {"content": [{"type": "tool_use", "id": "t2", "name": "get_media", "input": {}}], "stop_reason": "max_tokens"}


async def test_error_tool_results_are_flagged() -> None:
    model, sent = anthropic({"content": [{"type": "text", "text": "Not found."}], "stop_reason": "end_turn"})
    await model.chat(HISTORY, TOOLS)
    assert sent[0]["messages"][-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": '{"error": "media m_9 not found"}', "is_error": True}
    ]
    await model.aclose()


async def test_truncated_tool_call_is_retried_with_more_tokens() -> None:
    ok = {"content": [{"type": "tool_use", "id": "t2", "name": "get_media", "input": {"media_id": "m_1"}}], "stop_reason": "tool_use"}
    model, sent = anthropic(TRUNCATED, ok)
    resp = await model.chat(HISTORY, TOOLS, max_tokens=100)
    assert resp.tool_calls == [ToolCall("t2", "get_media", {"media_id": "m_1"})]
    assert [p["max_tokens"] for p in sent] == [100 * L.REASONING_BUDGET_FACTOR, 200 * L.REASONING_BUDGET_FACTOR]
    await model.aclose()


async def test_tool_call_truncated_twice_is_an_error() -> None:
    model, _ = anthropic(TRUNCATED, TRUNCATED)
    with pytest.raises(ProviderError, match="truncated"):
        await model.chat(HISTORY, TOOLS)
    await model.aclose()


async def test_truncated_text_reply_is_kept() -> None:
    model, sent = anthropic({"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"})
    assert (await model.chat(HISTORY, TOOLS)).text == "partial" and len(sent) == 1
    await model.aclose()


# ── images are encoded off the event loop ─────────────────────────────────────


async def test_images_are_encoded_in_a_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[str] = []
    real = L.encode_image

    def spy(img: Image.Image, *a: Any, **kw: Any) -> str:
        threads.append(threading.current_thread().name)
        return real(img, *a, **kw)

    monkeypatch.setattr(L, "encode_image", spy)
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "red"}, "finish_reason": "stop"}]})

    model = OpenAICompatModel("openai", "gpt-4.1", "https://api.openai.com/v1", "k", 10.0, 2)
    model._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    img = Image.new("RGB", (64, 64), (200, 0, 0))
    assert await model.complete([user("colour?", img, img)]) == "red"
    assert len(threads) == 1 and threads[0].startswith("ovp-cpu")  # once per image object, in the CPU pool
    parts = sent[0]["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"] and parts[1] == parts[2]
    await model.aclose()


# ── agent ─────────────────────────────────────────────────────────────────────


def test_system_prompt_asks_for_names_not_ids() -> None:
    assert "by name" in SYSTEM_PROMPT and "file name" in SYSTEM_PROMPT and "unnamed person" in SYSTEM_PROMPT
    assert "refer to items by id" not in SYSTEM_PROMPT


async def test_tool_failures_are_marked_as_errors(client: httpx.AsyncClient, app_ctx: Any) -> None:  # noqa: F811
    llm = ScriptedLLM(call("get_media", media_id="m_missing"), say("It does not exist."))
    app_ctx.registry.models["llm"] = llm
    await chat(client, "show m_missing")
    tool_msg = next(m for m in llm.requests[1] if m["role"] == "tool")
    assert tool_msg["is_error"] is True and "not found" in tool_msg["content"]


async def test_analyze_paths_reports_paths_outside_import_roots(client: httpx.AsyncClient, app_ctx: Any) -> None:  # noqa: F811
    llm = ScriptedLLM(call("analyze_paths", paths=["/etc"]), say("Not allowed."))
    app_ctx.registry.models["llm"] = llm
    events = await chat(client, "import /etc")
    result = next(e for e in events if e["type"] == "tool_result")
    tool_msg = next(m for m in llm.requests[1] if m["role"] == "tool")
    assert tool_msg["is_error"] is True and "OVP_IMPORT_ROOTS" in json.loads(tool_msg["content"])["error"]
    assert "OVP_IMPORT_ROOTS" in result["summary"]


async def test_agent_chat_rejects_unsafe_llm_spec(client: httpx.AsyncClient) -> None:  # noqa: F811
    r = await client.post("/api/agent/chat", json={"message": "hi", "llm": "qwen:/root/evil"})
    assert r.status_code == 400 and "working directory" in r.json()["detail"]
